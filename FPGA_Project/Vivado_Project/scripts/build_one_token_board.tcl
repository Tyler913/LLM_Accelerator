# Build and export the board-integrated one-token Qwen accelerator.
#
# The default action runs synthesis, implementation through write_bitstream,
# produces board-level reports, and exports a fixed XSA containing the bitstream.
#
# Examples:
#   vivado -mode batch -source scripts/build_one_token_board.tcl
#   vivado -mode batch -source scripts/build_one_token_board.tcl -tclargs --synth-only
#   vivado -mode batch -source scripts/build_one_token_board.tcl -tclargs --reuse-synth
#   vivado -mode batch -source scripts/build_one_token_board.tcl -tclargs --jobs=4 --out-dir=F:/tmp/qwen_board

set synth_only 0
set reuse_synth 0
set jobs 4
set requested_out_dir ""

for {set arg_index 0} {$arg_index < [llength $argv]} {incr arg_index} {
    set arg [lindex $argv $arg_index]
    if {$arg eq "--synth-only"} {
        set synth_only 1
    } elseif {$arg eq "--reuse-synth"} {
        set reuse_synth 1
    } elseif {$arg eq "--jobs"} {
        incr arg_index
        if {$arg_index >= [llength $argv]} {
            error "Missing value after --jobs"
        }
        set jobs [lindex $argv $arg_index]
    } elseif {[string match "--jobs=*" $arg]} {
        set jobs [string range $arg [string length "--jobs="] end]
    } elseif {$arg eq "--out-dir"} {
        incr arg_index
        if {$arg_index >= [llength $argv]} {
            error "Missing value after --out-dir"
        }
        set requested_out_dir [lindex $argv $arg_index]
    } elseif {[string match "--out-dir=*" $arg]} {
        set requested_out_dir [string range $arg [string length "--out-dir="] end]
    } else {
        error "Unknown argument '$arg'. Supported: --synth-only --reuse-synth --jobs=N --out-dir=PATH"
    }
}

if {![string is integer -strict $jobs] || $jobs < 1} {
    error "--jobs must be a positive integer"
}

set script_dir [file dirname [file normalize [info script]]]
set project_dir [file dirname $script_dir]
set project_file [file join $project_dir LLM_FPGA.xpr]
set bd_file [file join $project_dir LLM_FPGA.srcs sources_1 bd llm_system llm_system.bd]

if {$requested_out_dir eq ""} {
    set out_dir [file join $project_dir board_build one_token_board]
} else {
    set out_dir [file normalize $requested_out_dir]
}
file mkdir $out_dir

proc write_text_file {path contents} {
    set handle [open $path w]
    puts $handle $contents
    close $handle
}

proc write_run_status {out_dir run_name} {
    set run_object [get_runs $run_name]
    set status [get_property STATUS $run_object]
    set progress [get_property PROGRESS $run_object]
    set needs_refresh [get_property NEEDS_REFRESH $run_object]
    set summary [join [list \
        "run=$run_name" \
        "status=$status" \
        "progress=$progress" \
        "needs_refresh=$needs_refresh" \
    ] "\n"]
    write_text_file [file join $out_dir "${run_name}_status.txt"] $summary
    puts $summary
    return $status
}

proc require_completed_run {out_dir run_name} {
    set status [write_run_status $out_dir $run_name]
    set needs_refresh [get_property NEEDS_REFRESH [get_runs $run_name]]
    if {![string match -nocase "*complete*" $status] ||
        [string match -nocase "*error*" $status] ||
        $needs_refresh} {
        error "$run_name did not complete successfully: $status"
    }
}

proc open_llm_bd_with_roe_workaround {bd_file} {
    # One unrelated installed RoE Framer BD rule is unreadable on the current
    # Windows Vivado installation. The project does not use that IP, so skip
    # only that rule while opening the BD and immediately restore Tcl source.
    rename source qmap_board_build_original_source
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
        return [uplevel 1 [list qmap_board_build_original_source {*}$args]]
    }
    set open_status [catch {open_bd_design $bd_file} open_result open_options]
    rename source {}
    rename qmap_board_build_original_source source
    if {$open_status} {
        return -options $open_options $open_result
    }
}

open_project $project_file
open_llm_bd_with_roe_workaround $bd_file

set bd_object [get_files -quiet $bd_file]
if {[llength $bd_object] != 1} {
    error "Could not resolve exactly one llm_system BD: $bd_file"
}

validate_bd_design
save_bd_design
generate_target all $bd_object
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

if {!$reuse_synth} {
    # The legacy project had auto-incremental synthesis enabled against an
    # imported whole-design checkpoint.  That checkpoint can hide updated RTL
    # inside the module-reference OOC run and produce a stale utilization
    # report.  A board-readiness build must always rebuild the accelerator OOC
    # child and disable the imported incremental reference.
    set synth_run [get_runs synth_1]
    set_property AUTO_INCREMENTAL_CHECKPOINT 0 $synth_run
    set_property INCREMENTAL_CHECKPOINT "" $synth_run
    set qmap_ooc_runs [
        get_runs -quiet llm_system_qmap_one_token_axi_bd_0_0_synth_1
    ]
    if {[llength $qmap_ooc_runs] != 0} {
        reset_run $qmap_ooc_runs
    }
    reset_run synth_1
    launch_runs synth_1 -jobs $jobs
    wait_on_run synth_1
}
require_completed_run $out_dir synth_1

open_run synth_1
report_utilization \
    -file [file join $out_dir post_synth_utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 8 \
    -file [file join $out_dir post_synth_utilization_hierarchical.rpt]
report_timing_summary -delay_type max -max_paths 20 -report_unconstrained \
    -file [file join $out_dir post_synth_timing_summary.rpt]
report_methodology \
    -file [file join $out_dir post_synth_methodology.rpt]
report_drc \
    -file [file join $out_dir post_synth_drc.rpt]
close_design

if {$synth_only} {
    puts "Synthesis and post-synthesis reports completed successfully."
    close_project
    exit 0
}

# Never let a board-readiness export reuse an implementation generated from an
# older synthesized checkpoint.  In particular, --reuse-synth means "reuse the
# currently completed synth_1 result", not "reuse any existing impl_1 result".
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
require_completed_run $out_dir impl_1

open_run impl_1
report_utilization \
    -file [file join $out_dir post_route_utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 8 \
    -file [file join $out_dir post_route_utilization_hierarchical.rpt]
report_timing_summary -delay_type min_max -max_paths 50 -report_unconstrained \
    -file [file join $out_dir post_route_timing_summary.rpt]
check_timing -verbose \
    -file [file join $out_dir check_timing_verbose.rpt]
report_route_status \
    -file [file join $out_dir post_route_status.rpt]
report_clock_utilization \
    -file [file join $out_dir post_route_clock_utilization.rpt]
report_methodology \
    -file [file join $out_dir post_route_methodology.rpt]
report_drc \
    -file [file join $out_dir post_route_drc.rpt]
report_power \
    -file [file join $out_dir post_route_power.rpt]

set bit_candidates [glob -nocomplain \
    [file join $project_dir LLM_FPGA.runs impl_1 *.bit]]
if {[llength $bit_candidates] != 1} {
    error "Expected exactly one implementation bitstream, found [llength $bit_candidates]"
}
set bit_file [lindex $bit_candidates 0]
set exported_bit [file join $out_dir llm_system_qwen3_one_token_boardready.bit]
file copy -force $bit_file $exported_bit

set xsa_file [file join $out_dir llm_system_qwen3_one_token_boardready.xsa]
write_hw_platform -fixed -include_bit -force -file $xsa_file

write_text_file [file join $out_dir board_build_artifacts.txt] [join [list \
    "bitstream=$exported_bit" \
    "xsa=$xsa_file" \
] "\n"]

puts "Board bitstream: $exported_bit"
puts "Board hardware platform: $xsa_file"
close_design
close_project
