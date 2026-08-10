# configure_ps_gem3_network.tcl
#
# Safe-by-default Vivado configuration helper for the ALIENTEK/ATK MPSoC-P4
# PS Ethernet port. The board PS_ETH connector is wired to Zynq UltraScale+
# GEM3 through RGMII on MIO 64..75 and MDIO on MIO 76..77.
#
# Sourcing this file without arguments performs an audit only. The block design
# is changed and saved only when --apply is explicitly supplied. This script
# never launches synthesis, implementation, bitstream generation, simulation,
# output-product generation, or hardware export.
#
# Recommended use from an already-open Vivado GUI session:
#   1. Open LLM_FPGA.xpr and the llm_system block design.
#   2. source FPGA_Project/Vivado_Project/scripts/configure_ps_gem3_network.tcl
#   3. Review the audit output.
#   4. llm_ps_net::run --apply
#
# Static syntax check outside Vivado:
#   tclsh configure_ps_gem3_network.tcl --syntax-check

namespace eval llm_ps_net {
    variable script_dir [file dirname [file normalize [info script]]]
    variable expected_project_dir [file dirname $script_dir]
    variable expected_project_file [file join $expected_project_dir LLM_FPGA.xpr]
    variable expected_bd_file [file join $expected_project_dir LLM_FPGA.srcs sources_1 bd llm_system llm_system.bd]
    variable expected_project_name LLM_FPGA
    variable expected_part xczu2eg-sfvc784-2-i
    variable expected_bd_name llm_system
    variable expected_ps_cell zynq_ultra_ps_e_0
    variable expected_ps_vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5

    variable target_properties [list \
        CONFIG.PSU__ENET3__PERIPHERAL__ENABLE {1} \
        CONFIG.PSU__ENET3__PERIPHERAL__IO {MIO 64 .. 75} \
        CONFIG.PSU__ENET3__GRP_MDIO__ENABLE {1} \
        CONFIG.PSU__ENET3__GRP_MDIO__IO {MIO 76 .. 77} \
        CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__FREQMHZ {125} \
        CONFIG.PSU__TTC0__PERIPHERAL__ENABLE {1} \
        CONFIG.PSU__TTC0__PERIPHERAL__IO {NA} \
    ]

    variable audit_properties [list \
        CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__ACT_FREQMHZ \
        CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__SRCSEL \
        CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__DIVISOR0 \
        CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__DIVISOR1 \
        CONFIG.PSU__ENET3__PTP__ENABLE \
        CONFIG.PSU__ENET3__TSU__ENABLE \
        CONFIG.PSU_MIO_TREE_PERIPHERALS \
    ]
}

proc llm_ps_net::fail {message} {
    error "PS network configuration refused: $message"
}

proc llm_ps_net::require_exactly_one {objects description} {
    if {[llength $objects] != 1} {
        fail "expected exactly one $description, found [llength $objects]"
    }
    return [lindex $objects 0]
}

proc llm_ps_net::require_property {object property_name} {
    if {[lsearch -exact [list_property $object] $property_name] < 0} {
        fail "required property '$property_name' does not exist on [get_property NAME $object]"
    }
}

proc llm_ps_net::is_true {value} {
    return [expr {$value eq "1" || [string equal -nocase $value "true"] || [string equal -nocase $value "yes"]}]
}

proc llm_ps_net::is_125mhz {value} {
    if {[catch {expr {abs(double($value) - 125.0) <= 0.01}} result]} {
        return 0
    }
    return $result
}

proc llm_ps_net::values_match {property_name actual desired} {
    if {[string match "*FREQMHZ" $property_name]} {
        return [is_125mhz $actual]
    }
    return [expr {$actual eq $desired}]
}

proc llm_ps_net::require_allowed_value {object property_name allowed_values} {
    set value [get_property $property_name $object]
    if {[lsearch -exact $allowed_values $value] < 0} {
        fail "property '$property_name' has unexpected value '$value'; allowed pre-apply values are [join $allowed_values {, }]"
    }
}

proc llm_ps_net::assert_no_mio_64_77_conflict {ps_cell ignored_properties} {
    foreach property_name [lsort [list_property $ps_cell]] {
        if {![string match "CONFIG.*__IO" $property_name]} {
            continue
        }
        if {[lsearch -exact $ignored_properties $property_name] >= 0} {
            continue
        }

        set value [get_property $property_name $ps_cell]
        set low -1
        set high -1
        if {[regexp {^MIO[[:space:]]+([0-9]+)[[:space:]]+\.\.[[:space:]]+([0-9]+)$} $value -> low high]} {
            # Range parsed above.
        } elseif {[regexp {^MIO[[:space:]]+([0-9]+)$} $value -> low]} {
            set high $low
        } else {
            continue
        }

        if {$low <= 77 && $high >= 64} {
            fail "MIO 64..77 overlaps '$property_name'='$value'; resolve the pin assignment manually before enabling GEM3"
        }
    }
}

proc llm_ps_net::print_property_plan {ps_cell property_pairs heading} {
    puts $heading
    foreach {property_name desired} $property_pairs {
        set actual [get_property $property_name $ps_cell]
        if {[values_match $property_name $actual $desired]} {
            set state MATCH
        } else {
            set state CHANGE
        }
        puts [format {  %-58s %-18s -> %-18s [%s]} $property_name $actual $desired $state]
    }
}

proc llm_ps_net::print_clock_audit {ps_cell} {
    puts "Clock and optional-feature audit:"
    foreach property_name [list \
        CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__ACT_FREQMHZ \
        CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__SRCSEL \
        CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__DIVISOR0 \
        CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__DIVISOR1 \
        CONFIG.PSU__ENET3__PTP__ENABLE \
        CONFIG.PSU__ENET3__TSU__ENABLE \
    ] {
        puts [format "  %-58s %s" $property_name [get_property $property_name $ps_cell]]
    }
}

proc llm_ps_net::validate_context {} {
    variable expected_project_dir
    variable expected_project_file
    variable expected_bd_file
    variable expected_project_name
    variable expected_part
    variable expected_bd_name
    variable expected_ps_cell
    variable expected_ps_vlnv
    variable target_properties
    variable audit_properties

    set project [require_exactly_one [get_projects -quiet] "open Vivado project"]
    require_property $project DIRECTORY
    require_property $project IS_READONLY
    set project_name [get_property NAME $project]
    if {$project_name ne $expected_project_name} {
        fail "current project is '$project_name', expected '$expected_project_name'"
    }

    set project_part [get_property PART $project]
    if {$project_part ne $expected_part} {
        fail "current project part is '$project_part', expected '$expected_part'"
    }
    set project_dir [file normalize [get_property DIRECTORY $project]]
    if {$project_dir ne [file normalize $expected_project_dir]} {
        fail "current project directory is '$project_dir', expected '[file normalize $expected_project_dir]' from this script location"
    }
    if {![file isfile $expected_project_file]} {
        fail "expected project file is missing: $expected_project_file"
    }
    set bd_file_object [require_exactly_one [get_files -quiet $expected_bd_file] "project BD file '$expected_bd_file'"]
    require_property $bd_file_object IS_LOCKED

    set bd_design [current_bd_design -quiet]
    if {$bd_design eq ""} {
        fail "no block design is open; open '$expected_bd_name' before running this script"
    }
    set bd_name [get_property NAME $bd_design]
    if {$bd_name ne $expected_bd_name} {
        fail "current block design is '$bd_name', expected '$expected_bd_name'"
    }

    set ps_cell [require_exactly_one [get_bd_cells -quiet $expected_ps_cell] "BD cell '$expected_ps_cell'"]
    set ps_vlnv [get_property VLNV $ps_cell]
    if {$ps_vlnv ne $expected_ps_vlnv} {
        fail "cell '$expected_ps_cell' has VLNV '$ps_vlnv', expected '$expected_ps_vlnv'"
    }

    foreach {property_name desired} $target_properties {
        require_property $ps_cell $property_name
    }
    foreach property_name $audit_properties {
        require_property $ps_cell $property_name
    }

    require_allowed_value $ps_cell CONFIG.PSU__ENET3__PERIPHERAL__ENABLE [list 0 1]
    require_allowed_value $ps_cell CONFIG.PSU__ENET3__PERIPHERAL__IO [list "<Select>" "MIO 64 .. 75"]
    require_allowed_value $ps_cell CONFIG.PSU__ENET3__GRP_MDIO__ENABLE [list 0 1]
    require_allowed_value $ps_cell CONFIG.PSU__ENET3__GRP_MDIO__IO [list "<Select>" "MIO 76 .. 77"]
    require_allowed_value $ps_cell CONFIG.PSU__TTC0__PERIPHERAL__ENABLE [list 0 1]
    require_allowed_value $ps_cell CONFIG.PSU__TTC0__PERIPHERAL__IO [list NA]
    require_allowed_value $ps_cell CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__SRCSEL [list IOPLL]
    require_allowed_value $ps_cell CONFIG.PSU__ENET3__PTP__ENABLE [list 0]
    require_allowed_value $ps_cell CONFIG.PSU__ENET3__TSU__ENABLE [list 0]

    assert_no_mio_64_77_conflict $ps_cell [list \
        CONFIG.PSU__ENET3__PERIPHERAL__IO \
        CONFIG.PSU__ENET3__GRP_MDIO__IO \
    ]

    return [list $project $bd_design $bd_file_object $ps_cell]
}

proc llm_ps_net::assert_applied_values {ps_cell} {
    variable target_properties
    foreach {property_name desired} $target_properties {
        set actual [get_property $property_name $ps_cell]
        if {![values_match $property_name $actual $desired]} {
            fail "post-apply property '$property_name' is '$actual', expected '$desired'"
        }
    }

    set actual_frequency [get_property CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__ACT_FREQMHZ $ps_cell]
    if {![is_125mhz $actual_frequency]} {
        fail "computed GEM3 reference clock is '$actual_frequency' MHz, expected 125 MHz"
    }
}

proc llm_ps_net::run {args} {
    variable expected_project_file
    variable expected_bd_file
    variable expected_project_name
    variable expected_part
    variable expected_bd_name
    variable expected_ps_cell
    variable target_properties

    set apply_changes 0
    set syntax_check 0
    foreach arg $args {
        if {$arg eq "--apply"} {
            set apply_changes 1
        } elseif {$arg eq "--syntax-check"} {
            set syntax_check 1
        } else {
            fail "unknown argument '$arg'; supported arguments are --apply and --syntax-check"
        }
    }

    if {$syntax_check} {
        if {$apply_changes} {
            fail "--syntax-check and --apply cannot be used together"
        }
        puts "PASS Tcl syntax-load check; no Vivado commands were called."
        return
    }

    foreach required_command [list \
        get_projects get_files current_bd_design get_bd_cells list_property \
        get_property set_property validate_bd_design save_bd_design \
    ] {
        if {[llength [info commands $required_command]] == 0} {
            fail "Vivado command '$required_command' is unavailable; run this script inside Vivado"
        }
    }

    puts "PS GEM3/TTC0 configuration audit"
    puts "  project file:     $expected_project_file"
    puts "  BD file:          $expected_bd_file"
    puts "  expected project: $expected_project_name"
    puts "  expected part:    $expected_part"
    puts "  expected BD:      $expected_bd_name"
    puts "  expected PS cell: $expected_ps_cell"
    puts "  mode:             [expr {$apply_changes ? {APPLY AND SAVE} : {DRY RUN}}]"

    lassign [validate_context] project bd_design bd_file_object ps_cell
    set project_readonly [get_property IS_READONLY $project]
    set bd_locked [get_property IS_LOCKED $bd_file_object]
    puts "  project read-only: $project_readonly"
    puts "  BD file locked:   $bd_locked"

    print_property_plan $ps_cell $target_properties "Requested property plan:"
    print_clock_audit $ps_cell

    if {!$apply_changes} {
        puts "DRY RUN PASS: project, BD, PS cell, properties, current values, and MIO occupancy were audited."
        puts "NO CHANGES MADE. After reviewing this output, explicitly run: llm_ps_net::run --apply"
        return
    }

    if {[is_true $project_readonly]} {
        fail "the current project is read-only"
    }
    if {[is_true $bd_locked]} {
        fail "the llm_system BD file is locked"
    }

    set original_properties [list]
    foreach {property_name desired} $target_properties {
        lappend original_properties $property_name [get_property $property_name $ps_cell]
    }

    set save_started 0
    set apply_status [catch {
        set_property -dict $target_properties $ps_cell
        assert_applied_values $ps_cell
        print_property_plan $ps_cell $target_properties "Post-set property values before validation:"
        print_clock_audit $ps_cell

        puts "Validating llm_system before save..."
        validate_bd_design
        assert_applied_values $ps_cell

        puts "Saving llm_system.bd..."
        set save_started 1
        save_bd_design
    } apply_result apply_options]

    if {$apply_status} {
        puts stderr "APPLY FAILED: $apply_result"
        if {!$save_started} {
            set rollback_status [catch {set_property -dict $original_properties $ps_cell} rollback_result]
            if {$rollback_status} {
                puts stderr "ROLLBACK FAILED: $rollback_result"
                puts stderr "Close llm_system without saving and reopen it before continuing."
            } else {
                puts stderr "Rolled back the in-memory target properties; no save was attempted."
            }
        } else {
            puts stderr "The failure occurred while saving. Close llm_system without saving, reopen it, and audit the on-disk values before continuing."
        }
        return -options $apply_options $apply_result
    }

    puts "APPLY PASS: GEM3 RGMII/MDIO, 125 MHz GEM3 reference clock request, and TTC0 are configured and llm_system.bd is saved."
    puts "No synthesis, implementation, bitstream generation, simulation, output-product generation, or XSA export was launched."
}

if {[info exists ::argv]} {
    set llm_ps_net_startup_args $::argv
} else {
    set llm_ps_net_startup_args [list]
}
llm_ps_net::run {*}$llm_ps_net_startup_args
unset llm_ps_net_startup_args
