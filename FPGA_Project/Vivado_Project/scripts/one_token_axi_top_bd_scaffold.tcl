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

set apply_changes 0
set run_validate 0
set ctrl_base 0xA0040000
set ctrl_range 64K
set pl_ddr_base 0x0000000400000000
set pl_ddr_range 512M

foreach arg $argv {
    if {$arg eq "--apply"} {
        set apply_changes 1
    } elseif {$arg eq "--validate"} {
        set run_validate 1
    } elseif {[string match "--ctrl-base=*" $arg]} {
        set ctrl_base [string range $arg [string length "--ctrl-base="] end]
    } elseif {[string match "--ctrl-range=*" $arg]} {
        set ctrl_range [string range $arg [string length "--ctrl-range="] end]
    } elseif {[string match "--pl-ddr-base=*" $arg]} {
        set pl_ddr_base [string range $arg [string length "--pl-ddr-base="] end]
    } elseif {[string match "--pl-ddr-range=*" $arg]} {
        set pl_ddr_range [string range $arg [string length "--pl-ddr-range="] end]
    } else {
        error "Unknown argument '$arg'. Supported: --apply --validate --ctrl-base=0x... --ctrl-range=64K --pl-ddr-base=0x... --pl-ddr-range=512M"
    }
}

set script_dir [file dirname [file normalize [info script]]]
set project_dir [file dirname $script_dir]
set fpga_dir [file dirname $project_dir]
set rtl_dir [file join $fpga_dir rtl]
set project_file [file join $project_dir LLM_FPGA.xpr]
set bd_file [file join $project_dir LLM_FPGA.srcs sources_1 bd llm_system llm_system.bd]
set top_cell_name qmap_one_token_axi_top_0

set required_cells [list \
    zynq_ultra_ps_e_0 \
    axi_smc \
    rst_ps8_0_96M \
    ddr4_0 \
]

set required_rtl [list \
    axi4_read_master.sv \
    axi4_write_master.sv \
    axi4lite_to_mmio_regs.sv \
    q4_embedding_lookup.sv \
    qmap_one_token_control_regs.sv \
    qmap_one_token_top.sv \
    qmap_one_token_mmio_top.sv \
    qmap_one_token_axil_top.sv \
    qmap_one_token_axi_top.sv \
]

proc require_nonempty {objects message} {
    if {[llength $objects] == 0} {
        error $message
    }
    return $objects
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

puts "One-token AXI top BD scaffold"
puts "  project:    $project_file"
puts "  bd:         $bd_file"
puts "  rtl dir:    $rtl_dir"
puts "  ctrl base:  $ctrl_base / $ctrl_range"
puts "  PL DDR:     $pl_ddr_base / $pl_ddr_range"
puts "  apply:      $apply_changes"
puts "  validate:   $run_validate"

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
    puts "  4. Grow axi_smc to NUM_MI=6 and NUM_SI=3 when needed."
    puts "  5. Connect PS/HPM -> axi_smc -> $top_cell_name/S_AXI at $ctrl_base."
    puts "  6. Connect $top_cell_name/M_AXI -> axi_smc -> PL DDR at $pl_ddr_base."
    puts "  7. Optionally validate BD, then save."
    return
}

if {[llength [get_projects -quiet]] == 0} {
    open_project $project_file
}

set source_files [glob -nocomplain [file join $rtl_dir *.sv]]
add_files -norecurse -fileset sources_1 $source_files
foreach sv_file $source_files {
    set fobj [get_files -quiet $sv_file]
    if {[llength $fobj] != 0} {
        set_property FILE_TYPE SystemVerilog $fobj
    }
}
update_compile_order -fileset sources_1

open_bd_design $bd_file

foreach cell $required_cells {
    require_nonempty [get_bd_cells -quiet $cell] "Required BD cell '$cell' is missing; this script expects the current row1024 smoke BD baseline."
}

if {[llength [get_bd_cells -quiet $top_cell_name]] == 0} {
    create_bd_cell -type module -reference qmap_one_token_axi_top $top_cell_name
    puts "Created module-reference cell $top_cell_name"
} else {
    puts "Cell already exists: $top_cell_name"
}

# Current row1024 smoke BD uses axi_smc M00..M04 and S00..S01. Add one MI for
# the one-token S_AXI register aperture and one SI for the one-token M_AXI PL-DDR
# master path.
set_property CONFIG.NUM_MI 6 [get_bd_cells axi_smc]
set_property CONFIG.NUM_SI 3 [get_bd_cells axi_smc]

ensure_net_connection zynq_ultra_ps_e_0/pl_clk0 ${top_cell_name}/aclk
ensure_net_connection rst_ps8_0_96M/peripheral_aresetn ${top_cell_name}/aresetn
ensure_intf_connection ${top_cell_name}/S_AXI axi_smc/M05_AXI
ensure_intf_connection ${top_cell_name}/M_AXI axi_smc/S02_AXI

# Assign PS-visible control aperture. Keep old smoke GPIO apertures intact.
set ps_space [require_nonempty [get_bd_addr_spaces -quiet zynq_ultra_ps_e_0/Data] "Missing PS Data address space"]
set ctrl_seg [require_nonempty [get_bd_addr_segs -quiet ${top_cell_name}/S_AXI/Reg] "Missing one-token S_AXI Reg address segment"]
assign_bd_address -target_address_space $ps_space -offset $ctrl_base -range $ctrl_range $ctrl_seg -force
puts "Assigned PS control segment at $ctrl_base / $ctrl_range"

# Assign PL-DDR aperture to the one-token memory master.
set one_token_space [require_nonempty [get_bd_addr_spaces -quiet ${top_cell_name}/M_AXI] "Missing one-token M_AXI address space"]
set ddr_seg [require_nonempty [get_bd_addr_segs -quiet ddr4_0/C0_DDR4_MEMORY_MAP/C0_DDR4_ADDRESS_BLOCK] "Missing DDR4 address segment"]
assign_bd_address -target_address_space $one_token_space -offset $pl_ddr_base -range $pl_ddr_range $ddr_seg -force
puts "Assigned one-token M_AXI DDR segment at $pl_ddr_base / $pl_ddr_range"

if {$run_validate} {
    validate_bd_design
}

save_bd_design
puts "Saved BD. Synthesis/implementation were not launched by this script."
