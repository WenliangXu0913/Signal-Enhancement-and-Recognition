function [layers] = creatlayers_log_res4_ceshi3SE
lgraph = layerGraph();
tempLayers = imageInputLayer([65 65 1],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(4,"Name","fc")
    reluLayer("Name","relu")
    fullyConnectedLayer(64,"Name","fc_1")
    sigmoidLayer("Name","sigmoid")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication")
    convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_1")
    fullyConnectedLayer(4,"Name","fc_2")
    reluLayer("Name","relu_16")
    fullyConnectedLayer(64,"Name","fc_3")
    sigmoidLayer("Name","sigmoid_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_1")
    convolution2dLayer([3 3],64,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_2")
    fullyConnectedLayer(4,"Name","fc_4")
    reluLayer("Name","relu_17")
    fullyConnectedLayer(64,"Name","fc_5")
    sigmoidLayer("Name","sigmoid_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_2")
    convolution2dLayer([3 3],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_3")
    fullyConnectedLayer(4,"Name","fc_6")
    reluLayer("Name","relu_19")
    fullyConnectedLayer(64,"Name","fc_7")
    sigmoidLayer("Name","sigmoid_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_3")
    convolution2dLayer([3 3],64,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_4")
    fullyConnectedLayer(4,"Name","fc_8")
    reluLayer("Name","relu_20")
    fullyConnectedLayer(64,"Name","fc_9")
    sigmoidLayer("Name","sigmoid_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_4")
    convolution2dLayer([3 3],64,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_5")
    fullyConnectedLayer(4,"Name","fc_10")
    reluLayer("Name","relu_21")
    fullyConnectedLayer(64,"Name","fc_11")
    sigmoidLayer("Name","sigmoid_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_5")
    convolution2dLayer([3 3],64,"Name","conv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_6")
    fullyConnectedLayer(4,"Name","fc_12")
    reluLayer("Name","relu_22")
    fullyConnectedLayer(64,"Name","fc_13")
    sigmoidLayer("Name","sigmoid_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_6")
    convolution2dLayer([3 3],64,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_7")
    fullyConnectedLayer(4,"Name","fc_14")
    reluLayer("Name","relu_23")
    fullyConnectedLayer(64,"Name","fc_15")
    sigmoidLayer("Name","sigmoid_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_7")
    convolution2dLayer([3 3],64,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_8")
    fullyConnectedLayer(4,"Name","fc_16")
    reluLayer("Name","relu_24")
    fullyConnectedLayer(64,"Name","fc_17")
    sigmoidLayer("Name","sigmoid_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_8")
    convolution2dLayer([3 3],64,"Name","conv_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")
    reluLayer("Name","relu_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_9")
    fullyConnectedLayer(4,"Name","fc_18")
    reluLayer("Name","relu_25")
    fullyConnectedLayer(64,"Name","fc_19")
    sigmoidLayer("Name","sigmoid_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_9")
    convolution2dLayer([3 3],64,"Name","conv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")
    reluLayer("Name","relu_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_10")
    fullyConnectedLayer(4,"Name","fc_20")
    reluLayer("Name","relu_26")
    fullyConnectedLayer(64,"Name","fc_21")
    sigmoidLayer("Name","sigmoid_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_10")
    convolution2dLayer([3 3],64,"Name","conv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")
    reluLayer("Name","relu_12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_11")
    fullyConnectedLayer(4,"Name","fc_22")
    reluLayer("Name","relu_27")
    fullyConnectedLayer(64,"Name","fc_23")
    sigmoidLayer("Name","sigmoid_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_11")
    convolution2dLayer([3 3],64,"Name","conv_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_12")
    reluLayer("Name","relu_13")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_12")
    fullyConnectedLayer(4,"Name","fc_24")
    reluLayer("Name","relu_28")
    fullyConnectedLayer(64,"Name","fc_25")
    sigmoidLayer("Name","sigmoid_12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_12")
    convolution2dLayer([3 3],64,"Name","conv_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_13")
    reluLayer("Name","relu_14")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_13")
    fullyConnectedLayer(4,"Name","fc_26")
    reluLayer("Name","relu_29")
    fullyConnectedLayer(64,"Name","fc_27")
    sigmoidLayer("Name","sigmoid_13")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_13")
    convolution2dLayer([3 3],64,"Name","conv_15","Padding","same")
    batchNormalizationLayer("Name","batchnorm_14")
    reluLayer("Name","relu_15")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_14")
    fullyConnectedLayer(4,"Name","fc_28")
    reluLayer("Name","relu_30")
    fullyConnectedLayer(64,"Name","fc_29")
    sigmoidLayer("Name","sigmoid_14")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_14")
    convolution2dLayer([3 3],64,"Name","conv_16_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_1")
    reluLayer("Name","relu_16_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_15")
    fullyConnectedLayer(4,"Name","fc_30")
    reluLayer("Name","relu_31")
    fullyConnectedLayer(64,"Name","fc_31")
    sigmoidLayer("Name","sigmoid_15")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_15")
    convolution2dLayer([3 3],64,"Name","conv_16_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_2")
    reluLayer("Name","relu_16_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_16")
    fullyConnectedLayer(4,"Name","fc_32")
    reluLayer("Name","relu_32")
    fullyConnectedLayer(64,"Name","fc_33")
    sigmoidLayer("Name","sigmoid_16")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_16")
    convolution2dLayer([3 3],64,"Name","conv_16_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_3")
    reluLayer("Name","relu_16_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_17")
    fullyConnectedLayer(4,"Name","fc_34")
    reluLayer("Name","relu_33")
    fullyConnectedLayer(64,"Name","fc_35")
    sigmoidLayer("Name","sigmoid_17")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_17")
    convolution2dLayer([3 3],64,"Name","conv_16_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_4")
    reluLayer("Name","relu_16_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_18")
    fullyConnectedLayer(4,"Name","fc_36")
    reluLayer("Name","relu_34")
    fullyConnectedLayer(64,"Name","fc_37")
    sigmoidLayer("Name","sigmoid_18")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_18")
    convolution2dLayer([3 3],64,"Name","conv_16_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_5")
    reluLayer("Name","relu_16_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_19")
    fullyConnectedLayer(4,"Name","fc_38")
    reluLayer("Name","relu_35")
    fullyConnectedLayer(64,"Name","fc_39")
    sigmoidLayer("Name","sigmoid_19")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_19")
    convolution2dLayer([3 3],64,"Name","conv_16_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_6")
    reluLayer("Name","relu_16_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_20")
    fullyConnectedLayer(4,"Name","fc_40")
    reluLayer("Name","relu_36")
    fullyConnectedLayer(64,"Name","fc_41")
    sigmoidLayer("Name","sigmoid_20")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_20")
    convolution2dLayer([3 3],64,"Name","conv_16_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_7")
    reluLayer("Name","relu_16_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_21")
    fullyConnectedLayer(4,"Name","fc_42")
    reluLayer("Name","relu_37")
    fullyConnectedLayer(64,"Name","fc_43")
    sigmoidLayer("Name","sigmoid_21")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_21")
    convolution2dLayer([3 3],64,"Name","conv_16_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_8")
    reluLayer("Name","relu_16_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_22")
    fullyConnectedLayer(4,"Name","fc_44")
    reluLayer("Name","relu_38")
    fullyConnectedLayer(64,"Name","fc_45")
    sigmoidLayer("Name","sigmoid_22")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_22")
    convolution2dLayer([3 3],64,"Name","conv_16_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_9")
    reluLayer("Name","relu_16_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_23")
    fullyConnectedLayer(4,"Name","fc_46")
    reluLayer("Name","relu_39")
    fullyConnectedLayer(64,"Name","fc_47")
    sigmoidLayer("Name","sigmoid_23")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_23")
    convolution2dLayer([3 3],64,"Name","conv_16_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_10")
    reluLayer("Name","relu_16_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_24")
    fullyConnectedLayer(4,"Name","fc_48")
    reluLayer("Name","relu_40")
    fullyConnectedLayer(64,"Name","fc_49")
    sigmoidLayer("Name","sigmoid_24")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_24")
    convolution2dLayer([3 3],64,"Name","conv_16_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_11")
    reluLayer("Name","relu_16_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_25")
    fullyConnectedLayer(4,"Name","fc_50")
    reluLayer("Name","relu_41")
    fullyConnectedLayer(64,"Name","fc_51")
    sigmoidLayer("Name","sigmoid_25")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_25")
    convolution2dLayer([3 3],64,"Name","conv_16_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_12")
    reluLayer("Name","relu_16_12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_26")
    fullyConnectedLayer(4,"Name","fc_52")
    reluLayer("Name","relu_42")
    fullyConnectedLayer(64,"Name","fc_53")
    sigmoidLayer("Name","sigmoid_26")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_26")
    convolution2dLayer([3 3],64,"Name","conv_16_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_13")
    reluLayer("Name","relu_16_13")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_27")
    fullyConnectedLayer(4,"Name","fc_54")
    reluLayer("Name","relu_43")
    fullyConnectedLayer(64,"Name","fc_55")
    sigmoidLayer("Name","sigmoid_27")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_27")
    convolution2dLayer([3 3],64,"Name","conv_16_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_14")
    reluLayer("Name","relu_16_14")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_28")
    fullyConnectedLayer(4,"Name","fc_56")
    reluLayer("Name","relu_44")
    fullyConnectedLayer(64,"Name","fc_57")
    sigmoidLayer("Name","sigmoid_28")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_28")
    convolution2dLayer([3 3],64,"Name","conv_20","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu_18")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_29")
    fullyConnectedLayer(4,"Name","fc_58")
    reluLayer("Name","relu_45")
    fullyConnectedLayer(64,"Name","fc_59")
    sigmoidLayer("Name","sigmoid_29")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_29")
    convolution2dLayer([3 3],1,"Name","conv_19","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

lgraph = connectLayers(lgraph,"imageinput","conv_1");
lgraph = connectLayers(lgraph,"imageinput","addition/in2");
lgraph = connectLayers(lgraph,"relu_1","gapool");
lgraph = connectLayers(lgraph,"relu_1","multiplication/in2");
lgraph = connectLayers(lgraph,"sigmoid","multiplication/in1");
lgraph = connectLayers(lgraph,"relu_2","gapool_1");
lgraph = connectLayers(lgraph,"relu_2","multiplication_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1","multiplication_1/in1");
lgraph = connectLayers(lgraph,"relu_3","gapool_2");
lgraph = connectLayers(lgraph,"relu_3","multiplication_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_2","multiplication_2/in1");
lgraph = connectLayers(lgraph,"relu_4","gapool_3");
lgraph = connectLayers(lgraph,"relu_4","multiplication_3/in2");
lgraph = connectLayers(lgraph,"sigmoid_3","multiplication_3/in1");
lgraph = connectLayers(lgraph,"relu_5","gapool_4");
lgraph = connectLayers(lgraph,"relu_5","multiplication_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_4","multiplication_4/in1");
lgraph = connectLayers(lgraph,"relu_6","gapool_5");
lgraph = connectLayers(lgraph,"relu_6","multiplication_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_5","multiplication_5/in1");
lgraph = connectLayers(lgraph,"relu_7","gapool_6");
lgraph = connectLayers(lgraph,"relu_7","multiplication_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_6","multiplication_6/in1");
lgraph = connectLayers(lgraph,"relu_8","gapool_7");
lgraph = connectLayers(lgraph,"relu_8","multiplication_7/in2");
lgraph = connectLayers(lgraph,"sigmoid_7","multiplication_7/in1");
lgraph = connectLayers(lgraph,"relu_9","gapool_8");
lgraph = connectLayers(lgraph,"relu_9","multiplication_8/in2");
lgraph = connectLayers(lgraph,"sigmoid_8","multiplication_8/in1");
lgraph = connectLayers(lgraph,"relu_10","gapool_9");
lgraph = connectLayers(lgraph,"relu_10","multiplication_9/in2");
lgraph = connectLayers(lgraph,"sigmoid_9","multiplication_9/in1");
lgraph = connectLayers(lgraph,"relu_11","gapool_10");
lgraph = connectLayers(lgraph,"relu_11","multiplication_10/in2");
lgraph = connectLayers(lgraph,"sigmoid_10","multiplication_10/in1");
lgraph = connectLayers(lgraph,"relu_12","gapool_11");
lgraph = connectLayers(lgraph,"relu_12","multiplication_11/in2");
lgraph = connectLayers(lgraph,"sigmoid_11","multiplication_11/in1");
lgraph = connectLayers(lgraph,"relu_13","gapool_12");
lgraph = connectLayers(lgraph,"relu_13","multiplication_12/in2");
lgraph = connectLayers(lgraph,"sigmoid_12","multiplication_12/in1");
lgraph = connectLayers(lgraph,"relu_14","gapool_13");
lgraph = connectLayers(lgraph,"relu_14","multiplication_13/in2");
lgraph = connectLayers(lgraph,"sigmoid_13","multiplication_13/in1");
lgraph = connectLayers(lgraph,"relu_15","gapool_14");
lgraph = connectLayers(lgraph,"relu_15","multiplication_14/in2");
lgraph = connectLayers(lgraph,"sigmoid_14","multiplication_14/in1");
lgraph = connectLayers(lgraph,"relu_16_1","gapool_15");
lgraph = connectLayers(lgraph,"relu_16_1","multiplication_15/in2");
lgraph = connectLayers(lgraph,"sigmoid_15","multiplication_15/in1");
lgraph = connectLayers(lgraph,"relu_16_2","gapool_16");
lgraph = connectLayers(lgraph,"relu_16_2","multiplication_16/in2");
lgraph = connectLayers(lgraph,"sigmoid_16","multiplication_16/in1");
lgraph = connectLayers(lgraph,"relu_16_3","gapool_17");
lgraph = connectLayers(lgraph,"relu_16_3","multiplication_17/in2");
lgraph = connectLayers(lgraph,"sigmoid_17","multiplication_17/in1");
lgraph = connectLayers(lgraph,"relu_16_4","gapool_18");
lgraph = connectLayers(lgraph,"relu_16_4","multiplication_18/in2");
lgraph = connectLayers(lgraph,"sigmoid_18","multiplication_18/in1");
lgraph = connectLayers(lgraph,"relu_16_5","gapool_19");
lgraph = connectLayers(lgraph,"relu_16_5","multiplication_19/in2");
lgraph = connectLayers(lgraph,"sigmoid_19","multiplication_19/in1");
lgraph = connectLayers(lgraph,"relu_16_6","gapool_20");
lgraph = connectLayers(lgraph,"relu_16_6","multiplication_20/in2");
lgraph = connectLayers(lgraph,"sigmoid_20","multiplication_20/in1");
lgraph = connectLayers(lgraph,"relu_16_7","gapool_21");
lgraph = connectLayers(lgraph,"relu_16_7","multiplication_21/in2");
lgraph = connectLayers(lgraph,"sigmoid_21","multiplication_21/in1");
lgraph = connectLayers(lgraph,"relu_16_8","gapool_22");
lgraph = connectLayers(lgraph,"relu_16_8","multiplication_22/in2");
lgraph = connectLayers(lgraph,"sigmoid_22","multiplication_22/in1");
lgraph = connectLayers(lgraph,"relu_16_9","gapool_23");
lgraph = connectLayers(lgraph,"relu_16_9","multiplication_23/in2");
lgraph = connectLayers(lgraph,"sigmoid_23","multiplication_23/in1");
lgraph = connectLayers(lgraph,"relu_16_10","gapool_24");
lgraph = connectLayers(lgraph,"relu_16_10","multiplication_24/in2");
lgraph = connectLayers(lgraph,"sigmoid_24","multiplication_24/in1");
lgraph = connectLayers(lgraph,"relu_16_11","gapool_25");
lgraph = connectLayers(lgraph,"relu_16_11","multiplication_25/in2");
lgraph = connectLayers(lgraph,"sigmoid_25","multiplication_25/in1");
lgraph = connectLayers(lgraph,"relu_16_12","gapool_26");
lgraph = connectLayers(lgraph,"relu_16_12","multiplication_26/in2");
lgraph = connectLayers(lgraph,"sigmoid_26","multiplication_26/in1");
lgraph = connectLayers(lgraph,"relu_16_13","gapool_27");
lgraph = connectLayers(lgraph,"relu_16_13","multiplication_27/in2");
lgraph = connectLayers(lgraph,"sigmoid_27","multiplication_27/in1");
lgraph = connectLayers(lgraph,"relu_16_14","gapool_28");
lgraph = connectLayers(lgraph,"relu_16_14","multiplication_28/in2");
lgraph = connectLayers(lgraph,"sigmoid_28","multiplication_28/in1");
lgraph = connectLayers(lgraph,"relu_18","gapool_29");
lgraph = connectLayers(lgraph,"relu_18","multiplication_29/in2");
lgraph = connectLayers(lgraph,"sigmoid_29","multiplication_29/in1");
lgraph = connectLayers(lgraph,"conv_19","addition/in1");

layers=lgraph;
end