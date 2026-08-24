# Synchronize the Vivado sources_1 RTL entries with the tracked RTL manifests.
#
# Safe by default. Run from any directory:
#   vivado -mode batch -source FPGA_Project/Vivado_Project/scripts/sync_rtl_sources.tcl
#   vivado -mode batch -source FPGA_Project/Vivado_Project/scripts/sync_rtl_sources.tcl -tclargs --apply

set apply_changes 0
foreach arg $argv {
    if {$arg eq "--apply"} {
        set apply_changes 1
    } else {
        error "Unknown argument '$arg'. Supported: --apply"
    }
}

set script_dir [file dirname [file normalize [info script]]]
set project_dir [file dirname $script_dir]
set fpga_dir [file dirname $project_dir]
set rtl_dir [file normalize [file join $fpga_dir rtl]]
set project_file [file join $project_dir LLM_FPGA.xpr]
set source_manifest [file join $rtl_dir rtl_sources.list]
set include_manifest [file join $rtl_dir include_dirs.list]

proc path_is_under {path root} {
    set normalized_path [string tolower [file normalize $path]]
    set normalized_root [string trimright [string tolower [file normalize $root]] "/\\"]
    set prefix "${normalized_root}/"
    set slash_path [string map {\\ /} $normalized_path]
    set slash_prefix [string map {\\ /} $prefix]
    return [expr {[string first $slash_prefix $slash_path] == 0}]
}

proc read_relative_path_list {root list_file expected_kind} {
    if {![file exists $list_file]} {
        error "Missing RTL manifest: $list_file"
    }

    set handle [open $list_file r]
    set paths [list]
    set seen [dict create]
    while {[gets $handle line] >= 0} {
        set entry [string trim $line]
        if {$entry eq "" || [string index $entry 0] eq "#"} {
            continue
        }

        set path [file normalize [file join $root $entry]]
        if {![path_is_under $path $root]} {
            close $handle
            error "Manifest entry escapes its RTL root: $entry"
        }
        set path_key [string tolower [string map {\\ /} $path]]
        if {[dict exists $seen $path_key]} {
            close $handle
            error "Duplicate manifest entry: $entry"
        }
        dict set seen $path_key 1
        if {$expected_kind eq "file" && ![file isfile $path]} {
            close $handle
            error "RTL source entry does not exist: $path"
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

proc collect_files_recursive {root extensions} {
    set result [list]
    foreach entry [glob -nocomplain -directory $root *] {
        if {[file isdirectory $entry]} {
            set result [concat $result [collect_files_recursive $entry $extensions]]
        } elseif {[lsearch -exact -nocase $extensions [file extension $entry]] >= 0} {
            lappend result [file normalize $entry]
        }
    }
    return $result
}

proc normalized_path_set {paths} {
    set normalized [list]
    foreach path $paths {
        lappend normalized [string tolower [string map {\\ /} [file normalize $path]]]
    }
    return [lsort -dictionary -unique $normalized]
}

set source_files [read_relative_path_list $rtl_dir $source_manifest file]
set include_dirs [read_relative_path_list $rtl_dir $include_manifest directory]
foreach source_file $source_files {
    if {[lsearch -exact -nocase {.sv .v} [file extension $source_file]] < 0} {
        error "Unsupported RTL source extension: $source_file"
    }
}

set discovered_sources [collect_files_recursive $rtl_dir {.sv .v}]
if {[normalized_path_set $source_files] ne [normalized_path_set $discovered_sources]} {
    error "rtl_sources.list does not exactly cover the recursive .sv/.v source tree"
}

set header_files [list]
foreach include_dir $include_dirs {
    foreach header_file [collect_files_recursive $include_dir {.svh}] {
        if {[lsearch -exact -nocase $header_files $header_file] < 0} {
            lappend header_files [file normalize $header_file]
        }
    }
}
set discovered_headers [collect_files_recursive $rtl_dir {.svh}]
if {[normalized_path_set $header_files] ne [normalized_path_set $discovered_headers]} {
    error "include_dirs.list does not exactly cover the recursive .svh header tree"
}

set module_owners [dict create]
foreach source_file $source_files {
    set handle [open $source_file r]
    set contents [read $handle]
    close $handle
    set declarations [regexp -all -inline -line {^[ \t]*module[ \t]+([A-Za-z_][A-Za-z0-9_$]*)} $contents]
    if {[llength $declarations] != 2} {
        error "Expected exactly one module declaration in $source_file"
    }
    set module_name [lindex $declarations 1]
    if {[dict exists $module_owners $module_name]} {
        error "Duplicate module '$module_name' in $source_file and [dict get $module_owners $module_name]"
    }
    if {$module_name ne [file rootname [file tail $source_file]]} {
        error "Module '$module_name' does not match file name $source_file"
    }
    dict set module_owners $module_name $source_file
}

puts "RTL source synchronization"
puts "  project:  $project_file"
puts "  sources:  [llength $source_files]"
puts "  headers:  [llength $header_files]"
puts "  includes: [llength $include_dirs]"
puts "  apply:    $apply_changes"

if {!$apply_changes} {
    puts "DRY RUN ONLY: exact source/header coverage and module uniqueness are valid; no Vivado project changes were made."
    return
}

if {[llength [get_projects -quiet]] == 0} {
    open_project $project_file
}

set rtl_file_objects [list]
foreach file_object [get_files -quiet -of_objects [get_filesets sources_1]] {
    set file_name [get_property NAME $file_object]
    if {[path_is_under $file_name $rtl_dir]} {
        lappend rtl_file_objects $file_object
    }
}

if {[llength $rtl_file_objects] != 0} {
    remove_files -fileset sources_1 $rtl_file_objects
}

add_files -norecurse -fileset sources_1 $source_files
if {[llength $header_files] != 0} {
    add_files -norecurse -fileset sources_1 $header_files
}

foreach source_file $source_files {
    set file_object [get_files -quiet $source_file]
    if {[llength $file_object] != 0 &&
        [string equal -nocase [file extension $source_file] ".sv"] &&
        ![string equal -nocase [get_property FILE_TYPE $file_object] "SystemVerilog"]} {
        set_property FILE_TYPE SystemVerilog $file_object
    }
}

set_property include_dirs $include_dirs [get_filesets sources_1]
update_compile_order -fileset sources_1

set applied_include_dirs [get_property include_dirs [get_filesets sources_1]]
if {[normalized_path_set $applied_include_dirs] ne [normalized_path_set $include_dirs]} {
    error "Vivado sources_1 include_dirs does not match include_dirs.list after synchronization"
}

puts "Synchronized sources_1 with the tracked RTL manifests."
