`default_nettype none

// Sequential residual add with synchronous inferred block-RAM storage.
//
// This is the native-memory equivalent of residual_add_1024.  Both source
// vectors are loaded before i_start; the result remains available through a
// synchronous read port for a subsequent memory burst.
module residual_add_bram #(
    parameter int INPUT_SIZE       = 1024,
    parameter int RESIDUAL_WIDTH   = 24,
    parameter int RESIDUAL_FRAC    = 10,
    parameter int ADDEND_WIDTH     = 24,
    parameter int ADDEND_FRAC      = 12,
    parameter int OUT_WIDTH        = 24,
    parameter int OUT_FRAC         = 10,
    parameter int ELEMENT_INDEX_W  =
        (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE),
    parameter int ACC_WIDTH        = OUT_WIDTH + 2
)
(
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,

    input  wire logic                              i_residual_wr_en,
    input  wire logic [ELEMENT_INDEX_W-1 : 0]      i_residual_wr_addr,
    input  wire logic [RESIDUAL_WIDTH-1 : 0]       i_residual_wr_data,
    input  wire logic                              i_addend_wr_en,
    input  wire logic [ELEMENT_INDEX_W-1 : 0]      i_addend_wr_addr,
    input  wire logic [ADDEND_WIDTH-1 : 0]         i_addend_wr_data,

    input  wire logic                              i_start,
    output logic                                   o_busy,
    output logic                                   o_done,
    output logic                                   o_error,
    output logic                                   o_saturation,
    output logic [31 : 0]                          o_output_count,
    output logic [31 : 0]                          o_cycle_count,

    input  wire logic [ELEMENT_INDEX_W-1 : 0]      i_output_rd_addr,
    output logic [OUT_WIDTH-1 : 0]                 o_output_rd_data
);

    typedef enum logic [1 : 0] {
        S_IDLE,
        S_PRIME,
        S_RUN,
        S_DONE
    } state_t;

    localparam int ADDEND_TO_OUT_SHIFT =
        ADDEND_FRAC - OUT_FRAC;
    localparam logic [ELEMENT_INDEX_W-1 : 0] LAST_ELEMENT_INDEX =
        INPUT_SIZE - 1;
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MAX =
        {1'b0, {(OUT_WIDTH-1){1'b1}}};
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MIN =
        {1'b1, {(OUT_WIDTH-1){1'b0}}};
    localparam logic signed [ACC_WIDTH-1 : 0] OUT_MAX_EXT =
        {{(ACC_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX};
    localparam logic signed [ACC_WIDTH-1 : 0] OUT_MIN_EXT =
        {{(ACC_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN};

    state_t state;
    logic [ELEMENT_INDEX_W-1 : 0] element_index;
    logic [ELEMENT_INDEX_W-1 : 0] internal_rd_addr;

    (* ram_style = "block" *)
    logic signed [RESIDUAL_WIDTH-1 : 0]
        residual_mem [0 : INPUT_SIZE-1];
    (* ram_style = "block" *)
    logic signed [ADDEND_WIDTH-1 : 0]
        addend_mem [0 : INPUT_SIZE-1];
    (* ram_style = "block" *)
    logic signed [OUT_WIDTH-1 : 0]
        output_mem [0 : INPUT_SIZE-1];

    logic signed [RESIDUAL_WIDTH-1 : 0] residual_rd_data;
    logic signed [ADDEND_WIDTH-1 : 0] addend_rd_data;
    logic signed [ADDEND_WIDTH-1 : 0] addend_scaled_wide;
    logic signed [OUT_WIDTH-1 : 0] addend_scaled;
    logic signed [ACC_WIDTH-1 : 0] residual_ext;
    logic signed [ACC_WIDTH-1 : 0] addend_ext;
    logic signed [ACC_WIDTH-1 : 0] sum_ext;
    logic signed [OUT_WIDTH-1 : 0] saturated_sum;
    logic current_saturates;

    assign o_busy = (state == S_PRIME) || (state == S_RUN);
    assign o_done = (state == S_DONE);

    always_comb begin
        internal_rd_addr = element_index;
        if ((state == S_RUN) &&
            (element_index != LAST_ELEMENT_INDEX)) begin
            internal_rd_addr = element_index + 1'b1;
        end
    end

    // All vector memories are reset-free to preserve RAMB inference.
    always_ff @(posedge i_clk) begin
        if (i_residual_wr_en) begin
            residual_mem[i_residual_wr_addr] <= i_residual_wr_data;
        end
        residual_rd_data <= residual_mem[internal_rd_addr];

        if (i_addend_wr_en) begin
            addend_mem[i_addend_wr_addr] <= i_addend_wr_data;
        end
        addend_rd_data <= addend_mem[internal_rd_addr];

        if (state == S_RUN) begin
            output_mem[element_index] <= saturated_sum;
        end
        o_output_rd_data <= output_mem[i_output_rd_addr];
    end

    always_comb begin
        addend_scaled_wide =
            addend_rd_data >>> ADDEND_TO_OUT_SHIFT;
        addend_scaled = addend_scaled_wide[OUT_WIDTH-1 : 0];
        residual_ext =
            {{(ACC_WIDTH-RESIDUAL_WIDTH){
                residual_rd_data[RESIDUAL_WIDTH-1]}},
             residual_rd_data};
        addend_ext =
            {{(ACC_WIDTH-OUT_WIDTH){addend_scaled[OUT_WIDTH-1]}},
             addend_scaled};
        sum_ext = residual_ext + addend_ext;

        if (sum_ext > OUT_MAX_EXT) begin
            saturated_sum = OUT_MAX;
            current_saturates = 1'b1;
        end
        else if (sum_ext < OUT_MIN_EXT) begin
            saturated_sum = OUT_MIN;
            current_saturates = 1'b1;
        end
        else begin
            saturated_sum = sum_ext[OUT_WIDTH-1 : 0];
            current_saturates = 1'b0;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            element_index <= '0;
            o_error <= 1'b0;
            o_saturation <= 1'b0;
            o_output_count <= 32'd0;
            o_cycle_count <= 32'd0;
        end
        else begin
            if ((state == S_PRIME) || (state == S_RUN)) begin
                o_cycle_count <= o_cycle_count + 1'b1;
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        state <= S_PRIME;
                        element_index <= '0;
                        o_error <= 1'b0;
                        o_saturation <= 1'b0;
                        o_output_count <= 32'd0;
                        o_cycle_count <= 32'd0;
                    end
                end

                S_PRIME: begin
                    element_index <= '0;
                    state <= S_RUN;
                end

                S_RUN: begin
                    o_saturation <=
                        o_saturation | current_saturates;
                    o_output_count <= o_output_count + 1'b1;
                    if (element_index == LAST_ELEMENT_INDEX) begin
                        element_index <= '0;
                        state <= S_DONE;
                    end
                    else begin
                        element_index <= element_index + 1'b1;
                    end
                end

                S_DONE: begin
                    if (o_output_count != INPUT_SIZE) begin
                        o_error <= 1'b1;
                    end
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                    o_error <= 1'b1;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
