# one_token_axi_top_bd_scaffold.tcl
#
# Safe-by-default Vivado scaffold for adding the one-token AXI shell to the
# existing llm_system block design. By default this script is a dry-run and only
# prints the intended actions. Pass --apply to edit/save the BD. It intentionally
# does not launch synthesis/implementation.
#
# Usage examples from FPGA_Project/Vivado_Project:
#   vivado -mode batch -source scripts/one_token_axi_top_bd_scaffold.tcl
#   vivado -mode batch -source scripts/one_token_axi_top_bd_scaffold.tcl -tclargs --apply
#   vivado -mode batch -source scripts/one_token_axi_top_bd_scaffold.tcl -tclargs --apply --validate
#   vivado -mode batch -source scripts/one_token_axi_top_bd_scaffold.tcl -tclargs --apply --validate --generate-wrapper --mode=replace

set apply_changes 0
set run_validate 0
set generate_wrapper 0
set integration_mode augment
set ctrl_base 0xA0040000
set ctrl_range 64K
set pl_ddr_base 0x0000000400000000
set pl_ddr_range 512M
set ddr_status_base 0xA0010000
set ddr_status_range 64K

for {set arg_index 0} {$arg_index < [llength $argv]} {incr arg_index} {
    set arg [lindex $argv $arg_index]
    if {$arg eq "--apply"} {
        set apply_changes 1
    } elseif {$arg eq "--validate"} {
        set run_validate 1
    } elseif {$arg eq "--generate-wrapper"} {
        set generate_wrapper 1
    } elseif {$arg eq "--mode"} {
        incr arg_index
        if {$arg_index >= [llength $argv]} {
            error "Missing value after --mode"
        }
        set integration_mode [lindex $argv $arg_index]
    } elseif {[string match "--mode=*" $arg]} {
        set integration_mode [string range $arg [string length "--mode="] end]
    } elseif {$arg eq "--ctrl-base"} {
        incr arg_index
        if {$arg_index >= [llength $argv]} {
            error "Missing value after --ctrl-base"
        }
        set ctrl_base [lindex $argv $arg_index]
    } elseif {[string match "--ctrl-base=*" $arg]} {
        set ctrl_base [string range $arg [string length "--ctrl-base="] end]
    } elseif {$arg eq "--ctrl-range"} {
        incr arg_index
        if {$arg_index >= [llength $argv]} {
            error "Missing value after --ctrl-range"
        }
        set ctrl_range [lindex $argv $arg_index]
    } elseif {[string match "--ctrl-range=*" $arg]} {
        set ctrl_range [string range $arg [string length "--ctrl-range="] end]
    } elseif {$arg eq "--pl-ddr-base"} {
        incr arg_index
        if {$arg_index >= [llength $argv]} {
            error "Missing value after --pl-ddr-base"
        }
        set pl_ddr_base [lindex $argv $arg_index]
    } elseif {[string match "--pl-ddr-base=*" $arg]} {
        set pl_ddr_base [string range $arg [string length "--pl-ddr-base="] end]
    } elseif {$arg eq "--pl-ddr-range"} {
        incr arg_index
        if {$arg_index >= [llength $argv]} {
            error "Missing value after --pl-ddr-range"
        }
        set pl_ddr_range [lindex $argv $arg_index]
    } elseif {[string match "--pl-ddr-range=*" $arg]} {
        set pl_ddr_range [string range $arg [string length "--pl-ddr-range="] end]
    } else {
        error "Unknown argument '$arg'. Supported: --apply --validate --generate-wrapper --mode=augment|replace --ctrl-base=0x... --ctrl-range=64K --pl-ddr-base=0x... --pl-ddr-range=512M"
    }
}

if {$integration_mode ni {augment replace}} {
    error "Unsupported integration mode '$integration_mode'. Supported: augment replace"
}

set script_dir [file dirname [file normalize [info script]]]
set project_dir [file dirname $script_dir]
set fpga_dir [file dirname $project_dir]
set rtl_dir [file join $fpga_dir rtl]
set rtl_source_manifest [file join $rtl_dir rtl_sources.list]
set rtl_include_manifest [file join $rtl_dir include_dirs.list]
set project_file [file join $project_dir LLM_FPGA.xpr]
set bd_file [file join $project_dir LLM_FPGA.srcs sources_1 bd llm_system llm_system.bd]
set top_module_name qmap_one_token_axi_bd
set top_cell_name qmap_one_token_axi_bd_0

set retained_cells [list \
    zynq_ultra_ps_e_0 \
    axi_smc \
    rst_ps8_0_96M \
    ddr4_0 \
    proc_sys_reset_0 \
    axi_clock_converter_0 \
    axi_gpio_0 \
    xlconcat_0 \
    xlconstant_0 \
    xlconstant_1 \
]

set replace_smoke_cells [list \
    axi_bram_ctrl_0 \
    blk_mem_gen_0 \
    axi_gpio_1 \
    axi_gpio_2 \
    xlslice_0 \
    xlslice_1 \
    qmap_row1024_axi_smo_0 \
]

set required_cells $retained_cells
if {$integration_mode eq "augment"} {
    set required_cells [concat $required_cells $replace_smoke_cells]
}

set required_rtl [list \
    [file join lib bus axi4_read_master.sv] \
    [file join lib bus axi4_write_master.sv] \
    [file join lib bus axi4lite_to_mmio_regs.sv] \
    [file join lib memory qmap_base_addr_table_bram.sv] \
    [file join lib q4 q4_embedding_lookup.sv] \
    [file join top one_token qmap_one_token_control_regs.sv] \
    [file join top one_token qmap_one_token_top.sv] \
    [file join top one_token qmap_one_token_mmio_top.sv] \
    [file join top one_token qmap_one_token_axil_top.sv] \
    [file join top one_token qmap_one_token_axi_top.sv] \
    [file join top one_token qmap_one_token_axi_bd.v] \
]

proc require_nonempty {objects message} {
    if {[llength $objects] == 0} {
        error $message
    }
    return $objects
}

proc require_exactly_one {objects message} {
    if {[llength $objects] != 1} {
        error "$message (found [llength $objects])"
    }
    return [lindex $objects 0]
}

proc read_relative_path_list {root list_file expected_kind} {
    if {![file exists $list_file]} {
        error "Missing RTL manifest: $list_file"
    }

    set handle [open $list_file r]
    set paths [list]
    while {[gets $handle line] >= 0} {
        set entry [string trim $line]
        if {$entry eq "" || [string index $entry 0] eq "#"} {
            continue
        }

        set path [file normalize [file join $root $entry]]
        if {$expected_kind eq "file" && ![file isfile $path]} {
            close $handle
            error "RTL manifest file entry does not exist: $path"
        }
        if {$expected_kind eq "directory" && ![file isdirectory $path]} {
            close $handle
            error "RTL include directory does not exist: $path"
        }
        lappend paths $path
    }
    close $handle
    return $paths
}

proc ensure_intf_connection {src_pin dst_pin} {
    set src [require_nonempty [get_bd_intf_pins -quiet $src_pin] "Missing interface pin $src_pin"]
    set dst [require_nonempty [get_bd_intf_pins -quiet $dst_pin] "Missing interface pin $dst_pin"]
    set existing [get_bd_intf_nets -quiet -of_objects $src]
    if {[llength $existing] != 0} {
        set dst_nets [get_bd_intf_nets -quiet -of_objects $dst]
        if {[llength $dst_nets] != 0 && [string equal [get_property NAME $existing] [get_property NAME $dst_nets]]} {
            puts "Already connected: $src_pin -> $dst_pin"
            return
        }
        error "$src_pin is already connected to a different interface net"
    }
    connect_bd_intf_net $src $dst
    puts "Connected: $src_pin -> $dst_pin"
}

proc ensure_net_connection {src_pin dst_pin} {
    set src [require_nonempty [get_bd_pins -quiet $src_pin] "Missing pin $src_pin"]
    set dst [require_nonempty [get_bd_pins -quiet $dst_pin] "Missing pin $dst_pin"]
    set existing [get_bd_nets -quiet -of_objects $dst]
    if {[llength $existing] != 0} {
        puts "Destination already connected: $dst_pin"
        return
    }
    connect_bd_net $src $dst
    puts "Connected: $src_pin -> $dst_pin"
}

proc disconnect_intf_pin {pin_path} {
    set pins [get_bd_intf_pins -quiet $pin_path]
    if {[llength $pins] == 0} {
        return
    }
    set nets [get_bd_intf_nets -quiet -of_objects $pins]
    foreach net $nets {
        puts "Disconnecting interface net [get_property NAME $net] from $pin_path"
        delete_bd_objs $net
    }
}

proc delete_optional_bd_cell {cell_name} {
    set cells [get_bd_cells -quiet $cell_name]
    if {[llength $cells] == 0} {
        puts "Cell already absent: $cell_name"
        return
    }
    delete_bd_objs $cells
    puts "Deleted smoke-only cell: $cell_name"
}

proc delete_orphan_bd_nets {} {
    foreach net [get_bd_nets -quiet] {
        set endpoints [concat \
            [get_bd_pins -quiet -of_objects $net] \
            [get_bd_ports -quiet -of_objects $net]]
        if {[llength $endpoints] < 2} {
            puts "Deleting orphan scalar net [get_property NAME $net]"
            delete_bd_objs $net
        }
    }
    foreach net [get_bd_intf_nets -quiet] {
        set endpoints [concat \
            [get_bd_intf_pins -quiet -of_objects $net] \
            [get_bd_intf_ports -quiet -of_objects $net]]
        if {[llength $endpoints] < 2} {
            puts "Deleting orphan interface net [get_property NAME $net]"
            delete_bd_objs $net
        }
    }
}

proc rename_intf_net_for_pin {pin_path desired_name} {
    set pin [require_nonempty [get_bd_intf_pins -quiet $pin_path] "Missing interface pin $pin_path"]
    set net [require_exactly_one \
        [get_bd_intf_nets -quiet -of_objects $pin] \
        "Expected exactly one interface net on $pin_path"]
    if {[get_property NAME $net] ne $desired_name} {
        set_property NAME $desired_name $net
        puts "Renamed interface net on $pin_path to $desired_name"
    }
}

proc ensure_addr_seg_excluded {address_space slave_segment} {
    set address_space_path [get_property PATH $address_space]
    set excluded_segments [get_bd_addr_segs \
        -quiet -excluded -addressing -of_objects $slave_segment]
    foreach excluded_segment $excluded_segments {
        if {[string match "${address_space_path}/*" [get_property PATH $excluded_segment]]} {
            puts "Already excluded: $slave_segment from $address_space_path"
            return
        }
    }
    exclude_bd_addr_seg -target_address_space $address_space $slave_segment
}

puts "One-token AXI top BD scaffold"
puts "  project:    $project_file"
puts "  bd:         $bd_file"
puts "  rtl dir:    $rtl_dir"
puts "  sources:    $rtl_source_manifest"
puts "  includes:   $rtl_include_manifest"
puts "  mode:       $integration_mode"
puts "  ctrl base:  $ctrl_base / $ctrl_range"
puts "  PL DDR:     $pl_ddr_base / $pl_ddr_range"
puts "  DDR status: $ddr_status_base / $ddr_status_range"
puts "  apply:      $apply_changes"
puts "  validate:   $run_validate"
puts "  wrapper:    $generate_wrapper"

foreach f $required_rtl {
    set path [file join $rtl_dir $f]
    if {![file exists $path]} {
        error "Missing required RTL file: $path"
    }
}

if {!$apply_changes} {
    puts "DRY RUN ONLY: no project or BD changes will be made. Re-run with --apply to modify llm_system.bd."
    puts "Planned actions:"
    puts "  1. Open project and llm_system.bd."
    puts "  2. Add/sync SystemVerilog sources from $rtl_dir."
    puts "  3. Create module-reference cell $top_cell_name if absent."
    if {$integration_mode eq "replace"} {
        puts "  4. Delete smoke-only cells: [join $replace_smoke_cells {, }]."
        puts "  5. Keep DDR status/reset observability, use a 32-bit single-thread PS HPM port, and compact axi_smc to NUM_MI=3, NUM_SI=2."
        puts "  6. Connect PS/HPM -> axi_smc/M00 -> $top_cell_name/S_AXI at $ctrl_base."
        puts "  7. Connect $top_cell_name/M_AXI -> axi_smc/S01 -> PL DDR at $pl_ddr_base."
    } else {
        puts "  4. Grow axi_smc to NUM_MI=6 and NUM_SI=3 when needed."
        puts "  5. Connect PS/HPM -> axi_smc -> $top_cell_name/S_AXI at $ctrl_base."
        puts "  6. Connect $top_cell_name/M_AXI -> axi_smc -> PL DDR at $pl_ddr_base."
    }
    puts "  8. Optionally validate, save, and generate a top-level BD wrapper."
    return
}

if {[llength [get_projects -quiet]] == 0} {
    open_project $project_file
}

# Vivado 2025.1.1 on the current Windows installation tries to load the
# unrelated RoE Framer rule whenever any BD is opened.  Its installed helper
# Tcl is unreadable to Vivado even though this design does not instantiate RoE,
# which otherwise aborts open_bd_design before this script can touch the BD.
# Open before refreshing compile order so the base IP Integrator feature loads
# with its normal source command; then skip only the unused RoE rule.
rename source qmap_original_source
namespace eval ::xilinx.com_bd_rule_roe_framer {
    proc init {args} {
        return
    }
}
proc source {args} {
    set source_path [file normalize [lindex $args end]]
    if {[string match -nocase "*/data/rsb/rules/roe_framer/bd.tcl" $source_path]} {
        puts "Skipping unused RoE Framer BD rule: $source_path"
        return
    }
    return [uplevel 1 [list qmap_original_source {*}$args]]
}
set open_bd_status [catch {open_bd_design $bd_file} open_bd_result open_bd_options]
rename source {}
rename qmap_original_source source
if {$open_bd_status} {
    return -options $open_bd_options $open_bd_result
}

set source_files [read_relative_path_list $rtl_dir $rtl_source_manifest file]
set include_dirs [read_relative_path_list $rtl_dir $rtl_include_manifest directory]
set header_files [list]
foreach include_dir $include_dirs {
    foreach header_file [glob -nocomplain -types f [file join $include_dir *.svh]] {
        lappend header_files $header_file
    }
}

set missing_source_files [list]
foreach source_file [concat $source_files $header_files] {
    if {[llength [get_files -quiet $source_file]] == 0} {
        lappend missing_source_files $source_file
    }
}
if {[llength $missing_source_files] != 0} {
    add_files -norecurse -fileset sources_1 $missing_source_files
}
foreach sv_file $source_files {
    set fobj [get_files -quiet $sv_file]
    if {[llength $fobj] != 0 && [string equal -nocase [file extension $sv_file] ".sv"]} {
        set_property FILE_TYPE SystemVerilog $fobj
    }
}
set_property include_dirs $include_dirs [get_filesets sources_1]
update_compile_order -fileset sources_1

foreach cell $required_cells {
    require_nonempty [get_bd_cells -quiet $cell] "Required BD cell '$cell' is missing; this script expects the current row1024 smoke BD baseline."
}

if {[llength [get_bd_cells -quiet $top_cell_name]] == 0} {
    create_bd_cell -type module -reference $top_module_name $top_cell_name
    puts "Created module-reference cell $top_cell_name"
} else {
    puts "Cell already exists: $top_cell_name"
}

if {$integration_mode eq "replace"} {
    # A prior augment-mode run may have connected the one-token cell at M05/S02.
    # Disconnect only its two interfaces before compacting SmartConnect.
    disconnect_intf_pin ${top_cell_name}/S_AXI
    disconnect_intf_pin ${top_cell_name}/M_AXI

    foreach cell $replace_smoke_cells {
        delete_optional_bd_cell $cell
    }
    delete_orphan_bd_nets

    # The PS only uses this PL-facing port for 32-bit control/status and model
    # image loading. Reducing it from the old 128-bit/four-thread setting avoids
    # a large SmartConnect width/thread converter without changing addresses or
    # the accelerator's independent 32-bit PL-DDR master.
    set_property -dict [list \
        CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
        CONFIG.PSU__HPM0_FPD__NUM_READ_THREADS {1} \
        CONFIG.PSU__HPM0_FPD__NUM_WRITE_THREADS {1} \
    ] [get_bd_cells zynq_ultra_ps_e_0]

    # Retain M01 for PL DDR and M02 for DDR status GPIO. Reuse the now-free M00
    # for one-token control. Retain S00 for PS and reuse S01 for one-token DDR.
    set_property CONFIG.NUM_MI 3 [get_bd_cells axi_smc]
    set_property CONFIG.NUM_SI 2 [get_bd_cells axi_smc]

    ensure_intf_connection zynq_ultra_ps_e_0/M_AXI_HPM0_FPD axi_smc/S00_AXI
    ensure_intf_connection axi_smc/M01_AXI axi_clock_converter_0/S_AXI
    ensure_intf_connection axi_smc/M02_AXI axi_gpio_0/S_AXI
    ensure_intf_connection ${top_cell_name}/S_AXI axi_smc/M00_AXI
    ensure_intf_connection ${top_cell_name}/M_AXI axi_smc/S01_AXI
    rename_intf_net_for_pin ${top_cell_name}/M_AXI ${top_cell_name}_M_AXI
} else {
    # Current row1024 smoke BD uses axi_smc M00..M04 and S00..S01. Add one MI
    # for one-token control and one SI for the one-token PL-DDR master.
    set_property CONFIG.NUM_MI 6 [get_bd_cells axi_smc]
    set_property CONFIG.NUM_SI 3 [get_bd_cells axi_smc]
    ensure_intf_connection ${top_cell_name}/S_AXI axi_smc/M05_AXI
    ensure_intf_connection ${top_cell_name}/M_AXI axi_smc/S02_AXI
}

ensure_net_connection zynq_ultra_ps_e_0/pl_clk0 ${top_cell_name}/aclk
ensure_net_connection rst_ps8_0_96M/peripheral_aresetn ${top_cell_name}/aresetn

# Assign PS-visible control aperture.
set ps_space [require_nonempty [get_bd_addr_spaces -quiet zynq_ultra_ps_e_0/Data] "Missing PS Data address space"]
set ctrl_seg [require_exactly_one \
    [get_bd_addr_segs -quiet -of_objects [get_bd_intf_pins ${top_cell_name}/S_AXI]] \
    "Expected exactly one one-token S_AXI register address segment"]
assign_bd_address -target_address_space $ps_space -offset $ctrl_base -range $ctrl_range $ctrl_seg -force
puts "Assigned PS control segment at $ctrl_base / $ctrl_range"

# Assign PL-DDR aperture to the one-token memory master.
set one_token_space [require_nonempty [get_bd_addr_spaces -quiet ${top_cell_name}/M_AXI] "Missing one-token M_AXI address space"]
set ddr_seg [require_nonempty [get_bd_addr_segs -quiet ddr4_0/C0_DDR4_MEMORY_MAP/C0_DDR4_ADDRESS_BLOCK] "Missing DDR4 address segment"]
assign_bd_address -target_address_space $one_token_space -offset $pl_ddr_base -range $pl_ddr_range $ddr_seg -force
puts "Assigned one-token M_AXI DDR segment at $pl_ddr_base / $pl_ddr_range"

if {$integration_mode eq "replace"} {
    # Keep DDR status observable from PS while preserving the proven DDR base.
    set status_seg [require_nonempty [get_bd_addr_segs -quiet axi_gpio_0/S_AXI/Reg] "Missing retained DDR status GPIO address segment"]
    assign_bd_address -target_address_space $ps_space -offset $ddr_status_base -range $ddr_status_range $status_seg -force
    assign_bd_address -target_address_space $ps_space -offset $pl_ddr_base -range $pl_ddr_range $ddr_seg -force
    puts "Preserved PS DDR status segment at $ddr_status_base / $ddr_status_range"
    puts "Preserved PS DDR segment at $pl_ddr_base / $pl_ddr_range"
    set one_token_unused_seg_paths [list axi_gpio_0/S_AXI/Reg]
} else {
    set one_token_unused_seg_paths [list \
        axi_bram_ctrl_0/S_AXI/Mem0 \
        axi_gpio_0/S_AXI/Reg \
        axi_gpio_1/S_AXI/Reg \
        axi_gpio_2/S_AXI/Reg \
    ]
}

# SmartConnect makes every downstream slave reachable from each upstream
# master. Restrict the one-token M_AXI space to PL DDR only.
foreach unused_seg_path $one_token_unused_seg_paths {
    set unused_seg [require_nonempty \
        [get_bd_addr_segs -quiet $unused_seg_path] \
        "Missing reachable segment '$unused_seg_path' while excluding it from one-token M_AXI"]
    ensure_addr_seg_excluded $one_token_space $unused_seg
}
ensure_addr_seg_excluded $one_token_space $ctrl_seg

if {$integration_mode eq "augment"} {
    set row1024_space [require_nonempty \
        [get_bd_addr_spaces -quiet qmap_row1024_axi_smo_0/M_AXI] \
        "Missing row1024 M_AXI address space"]
    ensure_addr_seg_excluded $row1024_space $ctrl_seg
}

# The default SmartConnect automation inserts register slices on every entry
# and exit plus switchboard handshake pipelines.  At this design's 96 MHz PL
# clock those stages are unnecessary, and they cost several thousand LUT/FFs
# on the small XCZU2EG.  Validate once to materialize the SmartConnect topology,
# then apply the documented per-node area overrides.  Back-pressure remains
# fully supported; this only removes optional pipeline storage.
if {$integration_mode eq "replace"} {
    validate_bd_design
    set_property CONFIG.ADVANCED_PROPERTIES {
        __view__ {
            functional {
                S00_Entry {SUPPORTS_WRAP 0}
                S01_Entry {SUPPORTS_WRAP 0}
            }
            timing {
                S00_Entry {MMU_REGSLICE 0 TR_REGSLICE 0}
                S01_Entry {MMU_REGSLICE 0 TR_REGSLICE 0}
                S00_Buffer {
                    AR_M_SEND_PIPE 0
                    AW_M_SEND_PIPE 0
                    W_M_SEND_PIPE 0
                }
                S01_Buffer {
                    AR_M_SEND_PIPE 0
                    AW_M_SEND_PIPE 0
                    W_M_SEND_PIPE 0
                }
                M00_Buffer {B_M_SEND_PIPE 0 R_M_SEND_PIPE 0}
                M01_Buffer {B_M_SEND_PIPE 0 R_M_SEND_PIPE 0}
                M02_Buffer {B_M_SEND_PIPE 0 R_M_SEND_PIPE 0}
                M00_Exit {REGSLICE 0}
                M01_Exit {REGSLICE 0}
                M02_Exit {REGSLICE 0}
                SW0 {
                    AR_M_PIPE 0
                    AW_M_PIPE 0
                    B_M_PIPE 0
                    R_M_PIPE 0
                    W_M_PIPE 0
                }
            }
        }
    } [get_bd_cells axi_smc]
}

if {$run_validate} {
    validate_bd_design
}

save_bd_design
puts "Saved BD."

if {$generate_wrapper} {
    set bd_object [require_nonempty [get_files -quiet $bd_file] "Could not resolve saved BD file for wrapper generation"]
    set wrapper_files [make_wrapper -files $bd_object -top]
    set unregistered_wrapper_files [list]
    foreach wrapper_file $wrapper_files {
        if {[llength [get_files -quiet $wrapper_file]] == 0} {
            lappend unregistered_wrapper_files $wrapper_file
        }
    }
    if {[llength $unregistered_wrapper_files] != 0} {
        add_files -norecurse -fileset sources_1 $unregistered_wrapper_files
        puts "Generated and registered top-level BD wrapper: [join $unregistered_wrapper_files {, }]"
    } elseif {[llength $wrapper_files] != 0} {
        puts "Top-level BD wrapper is already registered: [join $wrapper_files {, }]"
    } else {
        puts "Top-level BD wrapper is already current."
    }
    update_compile_order -fileset sources_1
}

puts "Synthesis/implementation were not launched by this script."
