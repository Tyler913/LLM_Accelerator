`timescale 1ns/1ps
`default_nettype none

// Simple smoke test for q4_gemv_row_1024.
//
// The stimulus is q_proj row 0 from:
//
//   artifacts/test_vectors/qwen3_0p6b_q4_v0/qkv_layer0_last_token_q4.npz
//
// Run from the repository root:
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_q4_gemv_row_1024.vvp \
//     FPGA_Project/sim/tb_q4_gemv_row_1024.sv \
//     FPGA_Project/rtl/q4_gemv_row_1024.sv \
//     FPGA_Project/rtl/q4_dot_product_64.sv
//   vvp FPGA_Project/sim/tb_q4_gemv_row_1024.vvp
//
// The default waveform output is:
//
//   FPGA_Project/wave/q4_gemv_row_1024.vcd
//
// Override it with:
//
//   vvp FPGA_Project/sim/tb_q4_gemv_row_1024.vvp +wavefile=<path>

module tb_q4_gemv_row_1024;

    localparam int INPUT_SIZE    = 1024;
    localparam int GROUP_SIZE    = 64;
    localparam int GROUP_COUNT   = INPUT_SIZE / GROUP_SIZE;
    localparam int ACT_WIDTH     = 16;
    localparam int WEIGHT_WIDTH  = 4;
    localparam int SCALE_WIDTH   = 16;
    localparam int ROW_ACC_WIDTH = 48;

    localparam logic [INPUT_SIZE*ACT_WIDTH-1:0] TEST_ACTIVATION_FLAT = {
        256'h051c0368007e01bfff78fc1c015f0377fdb1fa05fe250120047ffc8cfef502ab,
        256'h02b4fd2e022ffeb30314fb7bff7403cc01dcff63fffb01a20133037dfc850207,
        256'h02e901a5fea906090064fed301c8fc2203f402220105fae1035c0236049bff89,
        256'hfec80148fc300062ffa0fd56fc75fc530164009101f0046b005102d9fe6c01d1,
        256'h03c8fd64042efe06011303f30481013801a102c7fe5efec80112006eff15039d,
        256'hfde5fe8c01d40031fcaffda6fde2ff1c029dffc20387fdc3fdc200c2ff9afbe1,
        256'hfe81fc46ff63fe6504010249014c05400132fd9ffdcc0034fdcafd98ff46fc3d,
        256'h03060244032bfe60fd3f02f803400243fd1a01d6ff21ffdf0311fdf402a6052d,
        256'hfcb8f883fcd4036efea9fea801a002040408fdf20026fee501750090035f024c,
        256'h0254037dfdb5fe5efe45fd0e02ca00c402c3031d020cfdaaff0ffe9003170134,
        256'hfa50fe9afd94fddf00b9033e019200fcfee0051c03e40209fda90368fc3ffbdb,
        256'hff0601ad0240fc16ffab01befd01fe6bff2b01affbff0042ffc30139011a01c3,
        256'hfcdb02180429fc4cfcccff87fd3c0253fd76024c0313fb8d017a00ed04210211,
        256'hfddc047a011e0099030f0211fde3fec200f3016cfe8f0247fd8efdad046a0296,
        256'h013efbbcfd3cfe5f0443f9b8ff23fea2fe95fc52013cfd45ffc1fd71fcabfde1,
        256'h01980009fe48fd79fd3300a004dd0291fe550323fd0700ee011301bcff6500ae,
        256'h0053fd91fdb105b90477fca4003a0309fe760050fd3bfcf0fdcd03f100abfe13,
        256'hfa78031600ea00f30038fa0f016efd8e01f3fee1fab8fe5500eafe76031bfdf6,
        256'hfe65024e0090fcf3fdcf03edfb14fdde00c7036e029e00ae03b9fea8042b0119,
        256'h01e400d4ffcc01df0350fdbcfb0e014500c501bf04e501dafd1c01dd01960218,
        256'hfd1efe2a017602d8ff91fcfafcdbfdbf00950517fef0fecffd12fe13fd71fe8c,
        256'h00c7feea010efd320214032a032cffc00218fcadfd1ffda4fd060269ffea032c,
        256'hff2203a2fcdb010fff6303fb026305a4fd6ffc8cffa20386ffb4010000500415,
        256'hffafffa1fa33036403a4013afdfd03dbfc02fbe4041c01810021fb2a02f6fea4,
        256'hfe8603b6fc77fbd2018002e2020301dc00cf03ee014ffcd5fe15023f027b036e,
        256'hfcf0010ffced0201fdd003320023fdb90358fca201aefd290002fcebfd66fdc5,
        256'hff20018f0368ff16005e009a024703f8feb5033dfe24022cfbf30448025d026e,
        256'hfbe5014bfbb2032f0371fba4fbaf017bfd350390ff870146fd43fef10194fe3c,
        256'hfb80fe18045c02a5fdd5fd06fde5fe4d00b7febbfe5dfaab0424fdf503c3fcea,
        256'hfc1d03e1012a00abff96030dff78ffad035302b9008a0373fc79fc4efe4efe92,
        256'h019d0150fdecfcc9fd7f03bc06ab0161001fff0502b3014103250141fe1902d3,
        256'hfe610284040afce602e302e2fcf4010c056102f5fedc00e80083fd06ff83ff20,
        256'hfe2c00b902aafdf40220fd5a02acff49018bfee20203033ffcdc040a014afd84,
        256'h034efe1e000000a7fe2703d9fc8cfe270214ff8e013404e7fc88fe69fe39035f,
        256'h01abfff5fd19fbeffdf4fe5b011d0263fe74fcd0fda603d8ffd8fc3dfff60119,
        256'h01f6fe78fc99fe3efef2fe0dfaeafd1ffd66fd7b0197ff3afee1fef2018a030d,
        256'h04a6fea6fd870183000f0342fe1b0322fdbcfcfbfe54ff45026b037f01edfa23,
        256'h025ffbdc080301d8fbf300d5fe4b00f5fda9fdad036f01f402760293ff3c01b6,
        256'h0275fe0efd78fceaff39026e005903fefd6600bd0306fed0feabfd96005b0375,
        256'h015b013d0229006d03720250fe770240fd6201e7016c0285fc78009b01810251,
        256'hff78fc8b037a0086fdd503c7040401f0fd1d0337fbf102a6019ef27ffc5e01eb,
        256'hfb63ffeaff96fc86036e021e020d026a0206021dfdcd052bfe0f0172fee700e2,
        256'h0274fcf6fd5aff280220fb110280ff94fe7902ecfb4105840251fdefff6ffe3f,
        256'h052800650140030f02caff28035d0279019605a7fd5bfb430506fcd004b00194,
        256'hfd36f75a0240006f046bfe7300e6febf02f103bd003d03effd5efe92027c04e8,
        256'hfc6afa2f03b0fb940331fdf50623fe0efafefdef047d017b01410244021bffdc,
        256'h0107fd9f02eafdfffc3bffb402e001cc040c02070692ff3efc6500f3fda30113,
        256'hff5cfc4801e2fe60fc08fe940235027cffb7016300bffcb600bdfd23fd5bff77,
        256'h013501b80088fba806a401f9fe25fe32fda700e101f400f201a6029100b5fecc,
        256'hfd06ff100419fddb02c00098fe4f02c400b0ffe4fe8602f2ff440112004aff85,
        256'h011b033f018400fa01dafc0b01ce00f1fd62007e042a03f60112049dfc9afd86,
        256'h01d3fcc30174fe09fe1901ea007300580126000204290010ffa9feff01e1005b,
        256'hfeb1fb4600a700f3ffc401e8fe000141fec00259011e013400bdfc17f9e101ee,
        256'hfbd700de0037febdfc26fde9feaafdf30385fedd033c00390352ffbb0118fca8,
        256'hfe7802b302ab024603ee02fe01b40398ff87024c01e702370350fd6dfceb01ab,
        256'hfede0262fefc044cfed30299027bff35015d0075fc03fe9e01af02d7009dffed,
        256'hfff0fbed013300e6ff63fd37fed2fc4ffef7fd92ff12016102bffd62005bfbbf,
        256'h0253ff7dfd6f00cbfc880aae071dfbaf0050f9f200ccfc9c005f01bb039fff50,
        256'hfddc03b9ff5e00c60e31003803e7ff26fc7c0446003cfc7d013a024a01ecfdad,
        256'hfda40287fb4d035cfe4cfead00240251fbd4fdf5fdd10109fc8a00f2046e024d,
        256'h01f40103ff9df636ff5c009402e9ff06f191fde100a3fd120162ffe70074ff07,
        256'hfe9e04b3066dfdc9fd1cfeebfb3a022dffa3fe230450fca2e91dfcda014806e3,
        256'h043606ee01bd02d4df3e026300de032704c5ffdfffff017afc5afef6fa10011e,
        256'h0669edf1ecdd06e50391f9eb05dffc81ea89fb7709f6fcfcebcbe96b0b5c01b8
    };

    localparam logic [INPUT_SIZE*WEIGHT_WIDTH-1:0] TEST_WEIGHT_PACKED = {
        256'hef320f03f60be3e14f2e607dfeff44fcf100f03edd211f0df040fff11ba0f11f,
        256'hfe0d51711b2e3f22331030310d3f5df01dc2f052f10413ee23d06b101f4afe53,
        256'h143d171f5bfe1cfa20f1001d236201105022e013efc430f1610feffec1ffdf41,
        256'h0ee1df1c2ddb6e0d14ed506059310d3b042b1c022f173afb41e112a53513e041,
        256'h10fcd2e012f24220f724d4fe102b40f122c3f1300105efc0fec021f0fee00c0f,
        256'h220b0ef00eece3f113321c1b214211cfcfa2211f3cdd001a127b0d5df0fe02f3,
        256'h124d00a01e0ca1e4f3eef24203073d35df4fd3e0d706433e2000e30dc461cc95,
        256'h2e0450263ee0446f04131e0cafc1e12422c1e7d2ecfad00e1f3ca0f1f340dc3f,
        256'hf3ced1216debef20b42dc02e20f0ec23d20d42eeb1131a0b2010071f03b0f1f0,
        256'h21cb301feef0ec0f31c20fe1d0fb60e13110e01ede101f2e1009d51f4f2f04d2,
        256'he5f00240bfe2e031421e02fbed11ed0e3ed40ec0d01f4fe24f0d3ef22c2ddf9c,
        256'h301dd2d1f03c2f4e1f662a24a14432d710010bdf22d30b6022cbd24ff0f3f1dd,
        256'h42fb03251322d02d7403ed01c113351eec4420111f0ce043bdde4e33d0f3f111,
        256'h1e0b4fdffef4dd0d0d20cf1f0fd0f0ef7ae153110eed3122134102de33d02209,
        256'h11f3310fff0522e22e9e010fe7e21e3e11e01c0fffd2dd2feffc2cf32f010300,
        256'h3913f0e3d30d1131efde3002fd1eff1e3503fc4f1012f1ed0fe150ee30023ef1
    };

    localparam logic [GROUP_COUNT*SCALE_WIDTH-1:0] TEST_SCALE_FLAT =
        256'h004500390044003b005600440039004400480053004900440045004a0063007a;

    localparam logic signed [ROW_ACC_WIDTH-1:0] EXPECTED_ROW_SUM_Q26 =
        -48'sd3482169;

    logic clk;
    logic rst_n;
    logic start;
    logic [INPUT_SIZE*ACT_WIDTH-1:0] activation_flat;
    logic [INPUT_SIZE*WEIGHT_WIDTH-1:0] weight_packed;
    logic [GROUP_COUNT*SCALE_WIDTH-1:0] scale_flat;
    logic busy;
    logic done;
    logic signed [ROW_ACC_WIDTH-1:0] row_sum_q26;

    string wavefile;
    int cycle_count;

    q4_gemv_row_1024 dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_activation_flat(activation_flat),
        .i_weight_packed(weight_packed),
        .i_scale_flat(scale_flat),
        .o_busy(busy),
        .o_done(done),
        .o_row_sum_q26(row_sum_q26)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/q4_gemv_row_1024.vcd";
        void'($value$plusargs("wavefile=%s", wavefile));
        $dumpfile(wavefile);
        $dumpvars(0, tb_q4_gemv_row_1024);
    end

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        activation_flat = TEST_ACTIVATION_FLAT;
        weight_packed = TEST_WEIGHT_PACKED;
        scale_flat = TEST_SCALE_FLAT;
        cycle_count = 0;

        repeat (3) @(posedge clk);

        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;

        @(negedge clk);
        start = 1'b0;

        while ((done != 1'b1) && (cycle_count < 120)) begin
            @(posedge clk);
            cycle_count++;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for done");
            $finish(1);
        end

        #1;
        $display("q4_gemv_row_1024 q_proj row 0 smoke test");
        $display("  row_sum_q26 = %0d, expected = %0d",
                 row_sum_q26, EXPECTED_ROW_SUM_Q26);

        if (row_sum_q26 !== EXPECTED_ROW_SUM_Q26) begin
            $display("FAIL: row_sum_q26 mismatch");
            $finish(1);
        end

        $display("PASS: q4_gemv_row_1024 q_proj row 0 vector matched.");
        $display("Waveform: %s", wavefile);

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
