# Standalone XSDB launcher for the packaged Qwen3-0.6B board smoke.
#
# Expected package layout:
#   launch_qwen3_board.tcl
#   hw/llm_system_qwen3_one_token_boardready.{bit,xsa}
#   sw/{fsbl,a_qctl,a_qmdl}.elf
#   runtime/load_pl_ddr_runtime.tcl
#
# Environment:
#   QOT_BOARD_MODE       model (default) or control
#   QOT_HW_SERVER_URL    tcp:127.0.0.1:3121 by default
#   QOT_DEVICE_FILTER    XSDB target filter; defaults to name =~ "PL"

proc require_file {path label} {
    if {![file isfile $path]} {
        error "$label is missing: $path"
    }
}

proc set_force_memory_access {enabled} {
    # XSDB documents the plural spelling. Keep the singular unique-prefix form
    # as a compatibility fallback for launch environments that log that form.
    if {[catch {configparams force-mem-accesses $enabled}]} {
        configparams force-mem-access $enabled
    }
}

proc select_unique_target {filter description} {
    if {[catch {
        targets -set -nocase -filter $filter
    } message]} {
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

proc require_magic {address label} {
    set expected 0x50414D51
    set actual [read_u32 $address]
    if {$actual != $expected} {
        error [format \
            "%s QMAP sentinel mismatch at 0x%016llX: got 0x%08X expected 0x%08X" \
            $label $address $actual $expected]
    }
    puts [format \
        "PASS %s QMAP sentinel at 0x%016llX = 0x%08X" \
        $label $address $actual]
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

set package_root [file dirname [file normalize [info script]]]
set bit_file [file join \
    $package_root hw llm_system_qwen3_one_token_boardready.bit]
set xsa_file [file join \
    $package_root hw llm_system_qwen3_one_token_boardready.xsa]
set fsbl_file [file join $package_root sw fsbl.elf]
set control_elf [file join $package_root sw a_qctl.elf]
set model_elf [file join $package_root sw a_qmdl.elf]
set runtime_loader [file join $package_root runtime load_pl_ddr_runtime.tcl]

set mode "model"
if {[info exists ::env(QOT_BOARD_MODE)] &&
    [string trim $::env(QOT_BOARD_MODE)] ne ""} {
    set mode [string tolower [string trim $::env(QOT_BOARD_MODE)]]
}
if {$mode ni {"model" "control"}} {
    error "QOT_BOARD_MODE must be model or control, got: $mode"
}

set server_url "tcp:127.0.0.1:3121"
if {[info exists ::env(QOT_HW_SERVER_URL)] &&
    [string trim $::env(QOT_HW_SERVER_URL)] ne ""} {
    set server_url [string trim $::env(QOT_HW_SERVER_URL)]
}

set device_filter {name =~ "PL"}
if {[info exists ::env(QOT_DEVICE_FILTER)] &&
    [string trim $::env(QOT_DEVICE_FILTER)] ne ""} {
    set device_filter [string trim $::env(QOT_DEVICE_FILTER)]
}

require_file $bit_file "bitstream"
require_file $xsa_file "hardware platform"
require_file $fsbl_file "FSBL"
if {$mode eq "model"} {
    require_file $model_elf "model smoke ELF"
    require_file $runtime_loader "PL-DDR runtime loader"
} else {
    require_file $control_elf "control smoke ELF"
}

puts "Qwen3 board launcher"
puts "  package: $package_root"
puts "  mode: $mode"
puts "  hw_server: $server_url"
puts "  device filter: {$device_filter}"

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
puts "PASS programmed bitstream"

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

if {$mode eq "model"} {
    wait_for_pl_ddr4 0x00000000A0010000 600 100
    source $runtime_loader
    require_all_qmap_headers
    dow $model_elf
    puts "Starting full28 persistent two-token model smoke..."
    puts "Watch the board UART at 115200 8N1 for the final PASS/FAIL result."
} else {
    dow $control_elf
    puts "Starting AXI-Lite no-memory control smoke..."
    puts "Watch the board UART at 115200 8N1 for the final PASS/FAIL result."
}

con
set_force_memory_access 0
puts "PASS application downloaded and running"
