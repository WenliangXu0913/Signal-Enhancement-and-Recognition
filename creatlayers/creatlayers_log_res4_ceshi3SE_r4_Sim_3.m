function [layers] = creatlayers_log_res4_ceshi3SE_r4_Sim_3
lgraph = layerGraph();

tempLayers = imageInputLayer([65 65 1],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(16,"Name","fc")
    reluLayer("Name","relu")
    fullyConnectedLayer(64,"Name","fc_1")
    sigmoidLayer("Name","sigmoid")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication")
    convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_2")
    convolution2dLayer([3 3],64,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_3")
    convolution2dLayer([3 3],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_4")
    convolution2dLayer([3 3],64,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_5")
    convolution2dLayer([3 3],64,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_6")
    convolution2dLayer([3 3],64,"Name","conv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_7")
    convolution2dLayer([3 3],64,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_8")
    convolution2dLayer([3 3],64,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_9")
    convolution2dLayer([3 3],64,"Name","conv_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")
    reluLayer("Name","relu_10")
    convolution2dLayer([3 3],64,"Name","conv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")
    reluLayer("Name","relu_11")
    convolution2dLayer([3 3],64,"Name","conv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")
    reluLayer("Name","relu_12")
    convolution2dLayer([3 3],64,"Name","conv_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_12")
    reluLayer("Name","relu_13")
    convolution2dLayer([3 3],64,"Name","conv_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_13")
    reluLayer("Name","relu_14")
    convolution2dLayer([3 3],64,"Name","conv_15","Padding","same")
    batchNormalizationLayer("Name","batchnorm_14")
    reluLayer("Name","relu_15")
    convolution2dLayer([3 3],64,"Name","conv_16_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_1")
    reluLayer("Name","relu_16_1")
    convolution2dLayer([3 3],64,"Name","conv_16_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_2")
    reluLayer("Name","relu_16_2")
    convolution2dLayer([3 3],64,"Name","conv_16_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_3")
    reluLayer("Name","relu_16_3")
    convolution2dLayer([3 3],64,"Name","conv_16_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_4")
    reluLayer("Name","relu_16_4")
    convolution2dLayer([3 3],64,"Name","conv_16_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_5")
    reluLayer("Name","relu_16_5")
    convolution2dLayer([3 3],64,"Name","conv_16_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_6")
    reluLayer("Name","relu_16_6")
    convolution2dLayer([3 3],64,"Name","conv_16_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_7")
    reluLayer("Name","relu_16_7")
    convolution2dLayer([3 3],64,"Name","conv_16_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_8")
    reluLayer("Name","relu_16_8")
    convolution2dLayer([3 3],64,"Name","conv_16_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_9")
    reluLayer("Name","relu_16_9")
    convolution2dLayer([3 3],64,"Name","conv_16_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_10")
    reluLayer("Name","relu_16_10")
    convolution2dLayer([3 3],64,"Name","conv_16_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_11")
    reluLayer("Name","relu_16_11")
    convolution2dLayer([3 3],64,"Name","conv_16_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_12")
    reluLayer("Name","relu_16_12")
    convolution2dLayer([3 3],64,"Name","conv_16_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_13")
    reluLayer("Name","relu_16_13")
    convolution2dLayer([3 3],64,"Name","conv_16_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_14")
    reluLayer("Name","relu_16_14")
    convolution2dLayer([3 3],64,"Name","conv_16_14_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_14_1")
    reluLayer("Name","relu_16_14_1")
    convolution2dLayer([3 3],64,"Name","conv_20","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu_18")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_29")
    fullyConnectedLayer(16,"Name","fc_58")
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
lgraph = connectLayers(lgraph,"relu_18","gapool_29");
lgraph = connectLayers(lgraph,"relu_18","multiplication_29/in2");
lgraph = connectLayers(lgraph,"sigmoid_29","multiplication_29/in1");
lgraph = connectLayers(lgraph,"conv_19","addition/in1");
layers=lgraph;
end