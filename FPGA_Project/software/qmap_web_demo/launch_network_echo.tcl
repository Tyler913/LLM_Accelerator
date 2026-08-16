# XSDB launcher for the isolated PS GEM3/lwIP echo gate.
#
# The PowerShell wrapper validates the Vitis manifest and exports these paths:
#   QWEB_NETWORK_BIT, QWEB_NETWORK_XSA, QWEB_NETWORK_FSBL,
#   QWEB_NETWORK_ECHO_ELF.

proc require_env_file {name label} {
    if {![info exists ::env($name)] || [string trim $::env($name)] eq ""} {
        error "$name must name the $label"
    }
    set path [file normalize [string trim $::env($name)]]
    if {![file isfile $path]} {
        error "$label is missing: $path"
    }
    return $path
}

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

set bit_file [require_env_file QWEB_NETWORK_BIT "network bitstream"]
set xsa_file [require_env_file QWEB_NETWORK_XSA "network XSA"]
set fsbl_file [require_env_file QWEB_NETWORK_FSBL "network FSBL"]
set echo_elf [require_env_file QWEB_NETWORK_ECHO_ELF "lwIP echo ELF"]

set server_url "tcp:127.0.0.1:3121"
if {[info exists ::env(QWEB_HW_SERVER_URL)] &&
    [string trim $::env(QWEB_HW_SERVER_URL)] ne ""} {
    set server_url [string trim $::env(QWEB_HW_SERVER_URL)]
}

set device_filter {name =~ "PL"}
if {[info exists ::env(QWEB_DEVICE_FILTER)] &&
    [string trim $::env(QWEB_DEVICE_FILTER)] ne ""} {
    set device_filter [string trim $::env(QWEB_DEVICE_FILTER)]
}

puts "QWEB network echo launcher"
puts "  bitstream: $bit_file"
puts "  XSA: $xsa_file"
puts "  FSBL: $fsbl_file"
puts "  application: $echo_elf"
puts "  hw_server: $server_url"

connect -url $server_url
catch {bpremove -all}

if {[info exists ::env(XILINX_VITIS)]} {
    set zynqmp_utils [file join \
        $::env(XILINX_VITIS) scripts vitis util zynqmp_utils.tcl]
    if {[file isfile $zynqmp_utils]} {
        source $zynqmp_utils
    }
}

select_unique_target {name =~ "APU*"} "APU"
rst -system
after 3000

select_unique_target $device_filter "PL device"
fpga -file $bit_file
puts "PASS programmed network bitstream"

select_unique_target {name =~ "PSU*"} "PSU"
catch {disable_pmu_gate}
select_unique_target {name =~ "APU*"} "APU"
loadhw -hw $xsa_file -mem-ranges [list \
    {0x80000000 0xBFFFFFFF} \
    {0x400000000 0x5FFFFFFFF} \
    {0x1000000000 0x7FFFFFFFFF}]
set_force_memory_access 1

select_unique_target {name =~ "*A53*#0"} "Cortex-A53 #0"
rst -processor
dow $fsbl_file
set fsbl_breakpoint [bpadd -addr &XFsbl_Exit]
set fsbl_status [catch {con -block -timeout 60} fsbl_message]
catch {bpremove $fsbl_breakpoint}
if {$fsbl_status} {
    set_force_memory_access 0
    error "FSBL did not reach XFsbl_Exit: $fsbl_message"
}
puts "PASS FSBL initialization"

select_unique_target {name =~ "*A53*#0"} "Cortex-A53 #0"
rst -processor
dow $echo_elf
con
set_force_memory_access 0
puts "PASS network echo application downloaded and running"

