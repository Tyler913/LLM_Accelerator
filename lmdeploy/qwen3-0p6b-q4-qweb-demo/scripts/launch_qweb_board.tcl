# XSDB launcher for the audited board-hosted Qwen Web demo.
#
# The PowerShell wrapper audits and exports exactly these files:
#   QWEB_NETWORK_BIT, QWEB_NETWORK_XSA, QWEB_NETWORK_FSBL,
#   QWEB_NETWORK_WEB_ELF, and QWEB_RUNTIME_LOADER.
#
# This launcher deliberately owns no artifact-discovery policy.  It programs
# the audited network hardware, runs its FSBL, loads the already board-accepted
# 61-segment Qwen runtime, checks all 281 QMAP headers, and only then starts
# a_qweb.

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

proc read_u32 {address} {
    set values [mrd -value -size w $address 1]
    if {[llength $values] != 1} {
        error "Expected one word at [format 0x%016llX $address], got: $values"
    }
    return [expr {wide([lindex $values 0]) & 0xFFFFFFFF}]
}

proc wait_for_pl_ddr4 {status_address attempts delay_ms} {
    set last_status 0
    for {set attempt 0} {$attempt < $attempts} {incr attempt} {
        if {![catch {read_u32 $status_address} status]} {
            set last_status $status
            # bit 0: calibration complete; bit 1: UI reset; bit 2: AXI resetn
            if {[expr {($status & 0x7) == 0x5}]} {
                puts [format \
                    "PASS PL DDR4 ready status=0x%08X after %d poll(s)" \
                    $status [expr {$attempt + 1}]]
                return $status
            }
        }
        after $delay_ms
    }
    error [format \
        "PL DDR4 did not become ready; last status=0x%08X after %d poll(s)" \
        $last_status $attempts]
}

proc require_all_qmap_headers {} {
    set expected 0x50414D51
    set qkv_base 0x0000000405000000
    set qkv_stride 0x0000000000840000
    set body_base 0x0000000417900000
    set body_stride 0x0000000000100000
    set body_packets {
        {input_norm 0x000A0000}
        {attn_frontend 0x00020000}
        {attn_score_value 0x00030000}
        {o_proj 0x00040000}
        {post_attn_norm 0x00050000}
        {mlp_gate_up 0x00060000}
        {mlp_silu_mul 0x00070000}
        {mlp_down 0x00080000}
        {mlp_residual_add 0x00090000}
    }
    set checked 0

    for {set layer 0} {$layer < 28} {incr layer} {
        set qkv_address [expr {
            wide($qkv_base) + wide($layer) * wide($qkv_stride)
        }]
        set actual [read_u32 $qkv_address]
        if {$actual != $expected} {
            error [format \
                "layer%d qkv QMAP header mismatch at 0x%016llX: 0x%08X" \
                $layer $qkv_address $actual]
        }
        incr checked

        set layer_body_base [expr {
            wide($body_base) + wide($layer) * wide($body_stride)
        }]
        foreach packet $body_packets {
            lassign $packet packet_name packet_offset
            set packet_address [expr {
                wide($layer_body_base) + wide($packet_offset)
            }]
            set actual [read_u32 $packet_address]
            if {$actual != $expected} {
                error [format \
                    "layer%d %s QMAP header mismatch at 0x%016llX: 0x%08X" \
                    $layer $packet_name $packet_address $actual]
            }
            incr checked
        }
    }

    set final_tail_address 0x0000000419500000
    set actual [read_u32 $final_tail_address]
    if {$actual != $expected} {
        error [format \
            "final tail QMAP header mismatch at 0x%016llX: 0x%08X" \
            $final_tail_address $actual]
    }
    incr checked
    if {$checked != 281} {
        error "Internal QMAP header count mismatch: $checked"
    }
    puts "PASS all 281 QMAP packet headers"
}

set bit_file [require_env_file QWEB_NETWORK_BIT "network bitstream"]
set xsa_file [require_env_file QWEB_NETWORK_XSA "network XSA"]
set fsbl_file [require_env_file QWEB_NETWORK_FSBL "network FSBL"]
set web_elf [require_env_file QWEB_NETWORK_WEB_ELF "a_qweb ELF"]
set runtime_loader [require_env_file QWEB_RUNTIME_LOADER "Qwen runtime loader"]

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

puts "QWEB full-chain board launcher"
puts "  bitstream: $bit_file"
puts "  XSA: $xsa_file"
puts "  FSBL: $fsbl_file"
puts "  runtime loader: $runtime_loader"
puts "  application: $web_elf"
puts "  hw_server: $server_url"

set force_memory_enabled 0
set launch_status [catch {
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
    puts "PASS programmed audited network bitstream"

    select_unique_target {name =~ "PSU*"} "PSU"
    catch {disable_pmu_gate}
    select_unique_target {name =~ "APU*"} "APU"
    loadhw -hw $xsa_file -mem-ranges [list \
        {0x80000000 0xBFFFFFFF} \
        {0x400000000 0x5FFFFFFFF} \
        {0x1000000000 0x7FFFFFFFFF}]
    set_force_memory_access 1
    set force_memory_enabled 1

    select_unique_target {name =~ "*A53*#0"} "Cortex-A53 #0"
    rst -processor
    dow $fsbl_file
    set fsbl_breakpoint [bpadd -addr &XFsbl_Exit]
    set fsbl_status [catch {con -block -timeout 60} fsbl_message]
    catch {bpremove $fsbl_breakpoint}
    if {$fsbl_status} {
        error "FSBL did not reach XFsbl_Exit: $fsbl_message"
    }
    puts "PASS FSBL initialization"

    select_unique_target {name =~ "*A53*#0"} "Cortex-A53 #0"
    rst -processor
    wait_for_pl_ddr4 0x00000000A0010000 600 100
    source $runtime_loader
    require_all_qmap_headers

    dow $web_elf
    puts "Starting board-hosted Qwen Web demo..."
    puts "Watch UART at 115200 8N1 for the exact QWEB READY URL."
    con
} launch_message launch_options]

if {$force_memory_enabled} {
    catch {set_force_memory_access 0}
}
if {$launch_status} {
    return -options $launch_options $launch_message
}
puts "PASS a_qweb downloaded and running after audited runtime load"

