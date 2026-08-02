`timescale 1ns/1ps
`default_nettype none

// Real-vector regression for q4_gemv_row_bram.
//
// The vectors are q_proj rows 0 and 1 from the existing layer-0 Q4 export.
// This test:
//   - fills activation elements and packed 32-bit weight/scale words through
//     the native ports;
//   - inserts deterministic gaps in every fill stream (no AXI/backpressure
//     dependency exists inside the row engine);
//   - runs row 0 twice without reloading to prove memory persistence;
//   - overwrites only weight/scale storage and runs row 1;
//   - independently reconstructs every 64-element partial and scaled Q26
//     contribution before comparing the exact final integer result.
module tb_q4_gemv_row_bram_real #(
    parameter int INPUT_SIZE = 1024
);

    localparam int REAL_ROWS       = 16;
    localparam int BASE_INPUT_SIZE = 1024;
    localparam int GROUP_SIZE      = 64;
    localparam int GROUP_COUNT     = INPUT_SIZE / GROUP_SIZE;
    localparam int BASE_GROUP_COUNT = BASE_INPUT_SIZE / GROUP_SIZE;
    localparam int VECTOR_REPEAT_COUNT = INPUT_SIZE / BASE_INPUT_SIZE;
    localparam int SOURCE_ACT_WIDTH = 16;
    localparam int ACT_WIDTH       = 24;
    localparam int WEIGHT_WIDTH    = 4;
    localparam int SCALE_WIDTH     = 16;
    localparam int EXPECTED_WIDTH  = 48;
    localparam int PRODUCT_WIDTH   = ACT_WIDTH + WEIGHT_WIDTH;
    localparam int PARTIAL_WIDTH   =
        PRODUCT_WIDTH + $clog2(GROUP_SIZE);
    localparam int SCALED_WIDTH    = PARTIAL_WIDTH + SCALE_WIDTH;
    localparam int ROW_ACC_WIDTH   =
        SCALED_WIDTH + $clog2(GROUP_COUNT) + 2;
    localparam int ACT_ADDR_WIDTH  = $clog2(INPUT_SIZE);
    localparam int WEIGHT_WORDS    = INPUT_SIZE / 8;
    localparam int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_WORDS);
    localparam int SCALE_WORDS     = GROUP_COUNT / 2;
    localparam int SCALE_ADDR_WIDTH = $clog2(SCALE_WORDS);
    localparam int MAX_RUN_CYCLES  = INPUT_SIZE + 32;

    logic clk;
    logic rst_n;

    logic act_wr_valid;
    logic act_wr_ready;
    logic [ACT_ADDR_WIDTH-1 : 0] act_wr_addr;
    logic signed [ACT_WIDTH-1 : 0] act_wr_data;

    logic weight_wr_valid;
    logic weight_wr_ready;
    logic [WEIGHT_ADDR_WIDTH-1 : 0] weight_wr_addr;
    logic [31 : 0] weight_wr_data;

    logic scale_wr_valid;
    logic scale_wr_ready;
    logic [SCALE_ADDR_WIDTH-1 : 0] scale_wr_addr;
    logic [31 : 0] scale_wr_data;

    logic start;
    logic busy;
    logic done;
    logic error;
    logic signed [ROW_ACC_WIDTH-1 : 0] row_sum_q26;

    logic [SOURCE_ACT_WIDTH-1 : 0]
        activation_vectors [0 : BASE_INPUT_SIZE-1];
    logic [WEIGHT_WIDTH-1 : 0]
        weight_vectors [0 : REAL_ROWS*BASE_INPUT_SIZE-1];
    logic [SCALE_WIDTH-1 : 0]
        scale_vectors [0 : REAL_ROWS*BASE_GROUP_COUNT-1];
    logic signed [EXPECTED_WIDTH-1 : 0]
        expected_vectors [0 : REAL_ROWS-1];

    string vector_dir;
    string vector_prefix;
    string wavefile;

    integer mismatch_count;
    integer completed_runs;
    integer run_cycle_count;
    integer invalid_weight_read_count;
    integer saw_last_word_nibble6;
    integer saw_last_row_element;

    q4_gemv_row_bram #(
        .INPUT_SIZE       (INPUT_SIZE),
        .GROUP_SIZE       (GROUP_SIZE),
        .GROUP_COUNT      (GROUP_COUNT),
        .ACT_WIDTH        (ACT_WIDTH),
        .WEIGHT_WIDTH     (WEIGHT_WIDTH),
        .SCALE_WIDTH      (SCALE_WIDTH),
        .PRODUCT_WIDTH    (PRODUCT_WIDTH),
        .PARTIAL_WIDTH    (PARTIAL_WIDTH),
        .SCALED_WIDTH     (SCALED_WIDTH),
        .ROW_ACC_WIDTH    (ROW_ACC_WIDTH),
        .ACT_ADDR_WIDTH   (ACT_ADDR_WIDTH),
        .WEIGHT_WORDS     (WEIGHT_WORDS),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .SCALE_WORDS      (SCALE_WORDS),
        .SCALE_ADDR_WIDTH (SCALE_ADDR_WIDTH)
    ) dut (
        .i_clk            (clk),
        .i_rst_n          (rst_n),
        .i_act_wr_valid   (act_wr_valid),
        .o_act_wr_ready   (act_wr_ready),
        .i_act_wr_addr    (act_wr_addr),
        .i_act_wr_data    (act_wr_data),
        .i_weight_wr_valid(weight_wr_valid),
        .o_weight_wr_ready(weight_wr_ready),
        .i_weight_wr_addr (weight_wr_addr),
        .i_weight_wr_data (weight_wr_data),
        .i_scale_wr_valid (scale_wr_valid),
        .o_scale_wr_ready (scale_wr_ready),
        .i_scale_wr_addr  (scale_wr_addr),
        .i_scale_wr_data  (scale_wr_data),
        .i_start          (start),
        .o_busy           (busy),
        .o_done           (done),
        .o_error          (error),
        .o_row_sum_q26    (row_sum_q26)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // This assertion is especially important for INPUT_SIZE=3072. Its
    // 384-word memory needs a 9-bit address, so the illegal address 384 does
    // not wrap and would otherwise be a real out-of-range BRAM access.
    always @(posedge clk) begin
        if (rst_n && busy) begin
            if ({1'b0, dut.weight_read_addr} >= WEIGHT_WORDS) begin
                $display(
                    "FAIL: out-of-range weight BRAM read addr=%0d depth=%0d",
                    dut.weight_read_addr,
                    WEIGHT_WORDS
                );
                invalid_weight_read_count = invalid_weight_read_count + 1;
                mismatch_count = mismatch_count + 1;
            end

            if ((dut.state == 3'd3) &&
                (dut.element_index == ACT_ADDR_WIDTH'(INPUT_SIZE - 2))) begin
                saw_last_word_nibble6 = saw_last_word_nibble6 + 1;
                if (dut.prefetch_next_weight_word !== 1'b0) begin
                    $display(
                        "FAIL: final weight word attempted an illegal prefetch"
                    );
                    mismatch_count = mismatch_count + 1;
                end
                if (dut.weight_read_addr !==
                    WEIGHT_ADDR_WIDTH'(WEIGHT_WORDS - 1)) begin
                    $display(
                        "FAIL: element %0d read weight word %0d expected %0d",
                        INPUT_SIZE - 2,
                        dut.weight_read_addr,
                        WEIGHT_WORDS - 1
                    );
                    mismatch_count = mismatch_count + 1;
                end
            end

            if ((dut.state == 3'd3) &&
                (dut.element_index == ACT_ADDR_WIDTH'(INPUT_SIZE - 1))) begin
                saw_last_row_element = saw_last_row_element + 1;
                if (dut.weight_read_addr !==
                    WEIGHT_ADDR_WIDTH'(WEIGHT_WORDS - 1)) begin
                    $display(
                        "FAIL: final element read weight word %0d expected %0d",
                        dut.weight_read_addr,
                        WEIGHT_WORDS - 1
                    );
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    end

    initial begin
        wavefile = "FPGA_Project/wave/q4_gemv_row_bram_real.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, tb_q4_gemv_row_bram_real);
    end

    task automatic load_vector_files;
        begin
            $readmemh(
                {vector_dir, "/", vector_prefix, "_activation.hex"},
                activation_vectors
            );
            $readmemh(
                {vector_dir, "/", vector_prefix, "_weight.hex"},
                weight_vectors
            );
            $readmemh(
                {vector_dir, "/", vector_prefix, "_scale.hex"},
                scale_vectors
            );
            $readmemh(
                {vector_dir, "/", vector_prefix, "_expected.hex"},
                expected_vectors
            );
        end
    endtask

    task automatic wait_for_fill_ready;
        begin
            while (!(act_wr_ready && weight_wr_ready && scale_wr_ready)) begin
                @(negedge clk);
            end
        end
    endtask

    task automatic fill_activation;
        integer element;
        begin
            wait_for_fill_ready();
            act_wr_valid = 1'b0;

            for (element = 0; element < INPUT_SIZE;
                 element = element + 1) begin
                if ((element != 0) && ((element % 37) == 0)) begin
                    @(negedge clk);
                    act_wr_valid = 1'b0;
                end

                @(negedge clk);
                act_wr_valid = 1'b1;
                act_wr_addr = ACT_ADDR_WIDTH'(element);
                act_wr_data =
                    {{(ACT_WIDTH-SOURCE_ACT_WIDTH){
                        activation_vectors[
                            element % BASE_INPUT_SIZE
                        ][SOURCE_ACT_WIDTH-1]
                     }},
                     activation_vectors[element % BASE_INPUT_SIZE]};
            end

            @(negedge clk);
            act_wr_valid = 1'b0;
            act_wr_addr = '0;
            act_wr_data = '0;
        end
    endtask

    task automatic fill_weight_row;
        input integer row;
        integer word_index;
        integer nibble_index;
        integer vector_index;
        reg [31 : 0] packed_word;
        begin
            wait_for_fill_ready();
            weight_wr_valid = 1'b0;

            for (word_index = 0; word_index < WEIGHT_WORDS;
                 word_index = word_index + 1) begin
                packed_word = '0;
                for (nibble_index = 0; nibble_index < 8;
                     nibble_index = nibble_index + 1) begin
                    vector_index =
                        (row * BASE_INPUT_SIZE) +
                        (((word_index * 8) + nibble_index) %
                         BASE_INPUT_SIZE);
                    packed_word[nibble_index*WEIGHT_WIDTH +: WEIGHT_WIDTH] =
                        weight_vectors[vector_index];
                end

                if ((word_index != 0) && ((word_index % 11) == 0)) begin
                    @(negedge clk);
                    weight_wr_valid = 1'b0;
                end

                @(negedge clk);
                weight_wr_valid = 1'b1;
                weight_wr_addr = WEIGHT_ADDR_WIDTH'(word_index);
                weight_wr_data = packed_word;
            end

            @(negedge clk);
            weight_wr_valid = 1'b0;
            weight_wr_addr = '0;
            weight_wr_data = '0;
        end
    endtask

    task automatic fill_scale_row;
        input integer row;
        integer word_index;
        integer scale_index;
        reg [31 : 0] packed_word;
        begin
            wait_for_fill_ready();
            scale_wr_valid = 1'b0;

            for (word_index = 0; word_index < SCALE_WORDS;
                 word_index = word_index + 1) begin
                scale_index =
                    (row * BASE_GROUP_COUNT) +
                    ((word_index * 2) % BASE_GROUP_COUNT);
                packed_word = {
                    scale_vectors[
                        (row * BASE_GROUP_COUNT) +
                        (((word_index * 2) + 1) % BASE_GROUP_COUNT)
                    ],
                    scale_vectors[scale_index]
                };

                if ((word_index != 0) && ((word_index % 3) == 0)) begin
                    @(negedge clk);
                    scale_wr_valid = 1'b0;
                end

                @(negedge clk);
                scale_wr_valid = 1'b1;
                scale_wr_addr = SCALE_ADDR_WIDTH'(word_index);
                scale_wr_data = packed_word;
            end

            @(negedge clk);
            scale_wr_valid = 1'b0;
            scale_wr_addr = '0;
            scale_wr_data = '0;
        end
    endtask

    task automatic calculate_partial_derived_expected;
        input integer row;
        output reg signed [ROW_ACC_WIDTH-1 : 0] modeled_row;
        integer group;
        integer item;
        integer vector_index;
        integer activation_value;
        integer weight_value;
        integer scale_value;
        reg signed [63 : 0] partial_value;
        reg signed [63 : 0] scaled_value;
        reg signed [63 : 0] row_value;
        begin
            row_value = 64'sd0;

            for (group = 0; group < GROUP_COUNT;
                 group = group + 1) begin
                partial_value = 64'sd0;
                for (item = 0; item < GROUP_SIZE;
                     item = item + 1) begin
                    vector_index = (group * GROUP_SIZE) + item;
                    activation_value =
                        $signed(
                            activation_vectors[
                                vector_index % BASE_INPUT_SIZE
                            ]
                        );
                    weight_value =
                        $signed(
                            weight_vectors[
                                (row * BASE_INPUT_SIZE) +
                                (vector_index % BASE_INPUT_SIZE)
                            ]
                        );
                    partial_value =
                        partial_value +
                        (activation_value * weight_value);
                end

                scale_value =
                    scale_vectors[
                        (row * BASE_GROUP_COUNT) +
                        (group % BASE_GROUP_COUNT)
                    ];
                scaled_value = partial_value * scale_value;
                row_value = row_value + scaled_value;

                $display(
                    "MODEL row=%0d group=%0d partial=%0d scale=%0d scaled_q26=%0d",
                    row,
                    group,
                    partial_value,
                    scale_value,
                    scaled_value
                );
            end

            modeled_row = row_value;
        end
    endtask

    task automatic run_and_check;
        input integer row;
        input integer run_number;
        reg signed [ROW_ACC_WIDTH-1 : 0] modeled_row;
        reg signed [ROW_ACC_WIDTH-1 : 0] file_expected_row;
        reg signed [63 : 0] repeated_file_expected;
        integer cycles;
        begin
            calculate_partial_derived_expected(row, modeled_row);
            repeated_file_expected =
                $signed(expected_vectors[row]) * VECTOR_REPEAT_COUNT;
            file_expected_row = repeated_file_expected;

            if (modeled_row !== file_expected_row) begin
                $display(
                    "FAIL: row %0d software partial model=%0d file_expected=%0d",
                    row,
                    modeled_row,
                    file_expected_row
                );
                mismatch_count = mismatch_count + 1;
            end

            wait_for_fill_ready();
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            cycles = 0;
            while (!done && (cycles < MAX_RUN_CYCLES)) begin
                @(negedge clk);
                cycles = cycles + 1;
            end

            if (!done) begin
                $display(
                    "FAIL: row %0d run %0d timed out after %0d cycles",
                    row,
                    run_number,
                    cycles
                );
                mismatch_count = mismatch_count + 1;
            end
            else begin
                #1;
                $display(
                    "RUN row=%0d run=%0d cycles=%0d observed=%0d expected=%0d",
                    row,
                    run_number,
                    cycles,
                    row_sum_q26,
                    modeled_row
                );

                if (row_sum_q26 !== modeled_row) begin
                    $display(
                        "FAIL: row %0d run %0d exact Q26 mismatch",
                        row,
                        run_number
                    );
                    mismatch_count = mismatch_count + 1;
                end
                else begin
                    completed_runs = completed_runs + 1;
                end
            end

            if (error) begin
                $display("FAIL: DUT o_error asserted");
                mismatch_count = mismatch_count + 1;
            end

            run_cycle_count = cycles;
            @(negedge clk);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        act_wr_valid = 1'b0;
        act_wr_addr = '0;
        act_wr_data = '0;
        weight_wr_valid = 1'b0;
        weight_wr_addr = '0;
        weight_wr_data = '0;
        scale_wr_valid = 1'b0;
        scale_wr_addr = '0;
        scale_wr_data = '0;
        start = 1'b0;
        mismatch_count = 0;
        completed_runs = 0;
        run_cycle_count = 0;
        invalid_weight_read_count = 0;
        saw_last_word_nibble6 = 0;
        saw_last_row_element = 0;

        vector_dir = "FPGA_Project/sim/vectors";
        vector_prefix = "q4_gemv_projection_1024_real";
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("prefix=%s", vector_prefix)) begin
        end

        load_vector_files();

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fill_activation();
        fill_weight_row(0);
        fill_scale_row(0);
        run_and_check(0, 1);

        // Run again without any fill writes. Memory contents must survive both
        // reset-free compute storage and the prior transaction.
        run_and_check(0, 2);

        // Keep activation storage, overwrite only the row-specific memories.
        fill_weight_row(1);
        fill_scale_row(1);
        run_and_check(1, 3);

        if ((invalid_weight_read_count != 0) ||
            (saw_last_word_nibble6 != 3) ||
            (saw_last_row_element != 3)) begin
            $display(
                "FAIL: boundary coverage invalid_reads=%0d nibble6=%0d last=%0d",
                invalid_weight_read_count,
                saw_last_word_nibble6,
                saw_last_row_element
            );
            mismatch_count = mismatch_count + 1;
        end

        if ((mismatch_count != 0) || (completed_runs != 3)) begin
            $display(
                "FAIL: q4_gemv_row_bram real regression mismatches=%0d completed=%0d",
                mismatch_count,
                completed_runs
            );
            $finish(1);
        end

        $display(
            "PASS: q4_gemv_row_bram INPUT_SIZE=%0d real-vector-derived rows matched across 3 runs; last cycles=%0d.",
            INPUT_SIZE,
            run_cycle_count
        );
        $display(
            "BOUNDARY PASS: final-word nibble6/7 visits=%0d/%0d invalid_weight_reads=%0d.",
            saw_last_word_nibble6,
            saw_last_row_element,
            invalid_weight_read_count
        );
        $display("Waveform: %s", wavefile);

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
