# Read-only MDIO discovery for the PS GEM3 interface after an application has
# initialized the MAC.  The script stops Cortex-A53 #0 before issuing Clause 22
# management reads, so the target application cannot race the XSDB accesses.

proc set_force_memory_access {enabled} {
    if {[catch {configparams force-mem-accesses $enabled}]} {
        configparams force-mem-access $enabled
    }
}

proc select_unique_target {filter description} {
    if {[catch {targets -set -nocase -filter $filter} message]} {
        error "Could not select exactly one $description target with {$filter}: $message"
    }
}

proc read_u32 {address} {
    set values [mrd -value -size w $address 1]
    if {[llength $values] != 1} {
        error "Expected one word at [format 0x%016llX $address], got: $values"
    }
    return [expr {wide([lindex $values 0]) & 0xFFFFFFFF}]
}

set gem_base 0xFF0E0000
set nwctrl [expr {$gem_base + 0x00}]
set nwsr [expr {$gem_base + 0x08}]
set phymntnc [expr {$gem_base + 0x34}]
set mdio_idle_mask 0x00000004
set mdio_enable_mask 0x00000010

proc mdio_read_clause22 {phy register} {
    global nwsr phymntnc mdio_idle_mask
    if {$phy < 0 || $phy > 31 || $register < 0 || $register > 31} {
        error "Clause 22 PHY/register address is out of range"
    }
    if {([read_u32 $nwsr] & $mdio_idle_mask) == 0} {
        error "GEM3 MDIO controller was busy before read"
    }
    set command [expr {
        wide(0x60020000) | (wide($phy) << 23) | (wide($register) << 18)
    }]
    mwr -force $phymntnc $command
    for {set poll 0} {$poll < 1000} {incr poll} {
        if {([read_u32 $nwsr] & $mdio_idle_mask) != 0} {
            return [expr {[read_u32 $phymntnc] & 0xFFFF}]
        }
    }
    error "GEM3 MDIO read timed out for PHY $phy register $register"
}

proc mdio_write_clause22 {phy register value} {
    global nwsr phymntnc mdio_idle_mask
    if {$phy < 0 || $phy > 31 || $register < 0 || $register > 31 ||
        $value < 0 || $value > 0xFFFF} {
        error "Clause 22 PHY/register write is out of range"
    }
    if {([read_u32 $nwsr] & $mdio_idle_mask) == 0} {
        error "GEM3 MDIO controller was busy before write"
    }
    set command [expr {
        wide(0x50020000) | (wide($phy) << 23) |
        (wide($register) << 18) | wide($value)
    }]
    mwr -force $phymntnc $command
    for {set poll 0} {$poll < 1000} {incr poll} {
        if {([read_u32 $nwsr] & $mdio_idle_mask) != 0} {
            return
        }
    }
    error "GEM3 MDIO write timed out for PHY $phy register $register"
}

proc yt8521_read_extended {phy register} {
    # Register 0x1E is an address latch rather than a persistent page switch.
    # Restore its prior value so this diagnostic leaves the PHY configuration
    # unchanged.
    set prior [mdio_read_clause22 $phy 0x1E]
    mdio_write_clause22 $phy 0x1E $register
    set value [mdio_read_clause22 $phy 0x1F]
    mdio_write_clause22 $phy 0x1E $prior
    return $value
}

set server_url "tcp:127.0.0.1:3121"
if {[info exists ::env(QWEB_HW_SERVER_URL)] &&
    [string trim $::env(QWEB_HW_SERVER_URL)] ne ""} {
    set server_url [string trim $::env(QWEB_HW_SERVER_URL)]
}

connect -url $server_url
select_unique_target {name =~ "*A53*#0"} "Cortex-A53 #0"
catch {stop}
set_force_memory_access 1

set control [read_u32 $nwctrl]
set status [read_u32 $nwsr]
puts [format "GEM3 NWCTRL=0x%08X NWSR=0x%08X" $control $status]
if {($control & $mdio_enable_mask) == 0} {
    set_force_memory_access 0
    error "GEM3 MDIO port is not enabled"
}

set found 0
for {set phy 0} {$phy < 32} {incr phy} {
    set bmsr [mdio_read_clause22 $phy 1]
    set id1 [mdio_read_clause22 $phy 2]
    set id2 [mdio_read_clause22 $phy 3]
    if {(($bmsr != 0x0000) && ($bmsr != 0xFFFF)) ||
        (($id1 != 0x0000) && ($id1 != 0xFFFF)) ||
        (($id2 != 0x0000) && ($id2 != 0xFFFF))} {
        set bmcr [mdio_read_clause22 $phy 0]
        puts [format \
            "PHY_CANDIDATE addr=%d bmcr=0x%04X bmsr=0x%04X id1=0x%04X id2=0x%04X" \
            $phy $bmcr $bmsr $id1 $id2]
        incr found
    }
}

set_force_memory_access 0
if {$found == 0} {
    error "No Clause 22 PHY responded on GEM3 MDIO"
}
set yt_id1 [mdio_read_clause22 7 2]
set yt_id2 [mdio_read_clause22 7 3]
if {$yt_id1 == 0x0000 && $yt_id2 == 0x011A} {
    set specific [mdio_read_clause22 7 0x11]
    set reg_space [yt8521_read_extended 7 0xA000]
    set chip_config [yt8521_read_extended 7 0xA001]
    set rgmii_config [yt8521_read_extended 7 0xA003]
    set sleep_control [yt8521_read_extended 7 0x0027]
    puts [format \
        "YT8521 addr=7 specific=0x%04X reg_space=0x%04X chip_config=0x%04X rgmii_config=0x%04X sleep_control=0x%04X" \
        $specific $reg_space $chip_config $rgmii_config $sleep_control]
}
puts "PASS discovered $found GEM3 Clause 22 PHY candidate(s)"
