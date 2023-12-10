function [layers] = creatlayers_log_res4_ceshi3CBAM_r4_Sim_1_Res2
lgraph = layerGraph();

tempLayers = [
    imageInputLayer([65 65 1],"Name","imageinput")
    convolution2dLayer([3 3],64,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
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
    reluLayer("Name","relu_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(16,"Name","fc")
    reluLayer("Name","relu")
    fullyConnectedLayer(64,"Name","fc_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalMaxPooling2dLayer("Name","gmpool")
    fullyConnectedLayer(16,"Name","fc_2")
    reluLayer("Name","relu_13")
    fullyConnectedLayer(64,"Name","fc_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    sigmoidLayer("Name","sigmoid")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)mean(X,3),"Name","avgpool2d");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)max(X,[],3),"Name","maxpool");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat")
    convolution2dLayer([3 3],1,"Name","conv","Padding","same")
    sigmoidLayer("Name","sigmoid_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
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
    reluLayer("Name","relu_7_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_1")
    fullyConnectedLayer(16,"Name","fc_4")
    reluLayer("Name","relu_16")
    fullyConnectedLayer(64,"Name","fc_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalMaxPooling2dLayer("Name","gmpool_1")
    fullyConnectedLayer(16,"Name","fc_6")
    reluLayer("Name","relu_17")
    fullyConnectedLayer(64,"Name","fc_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_6")
    sigmoidLayer("Name","sigmoid_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)mean(X,3),"Name","avgpool2d_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)max(X,[],3),"Name","maxpool_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_1")
    convolution2dLayer([3 3],1,"Name","conv_16","Padding","same")
    sigmoidLayer("Name","sigmoid_6_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
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
    reluLayer("Name","relu_7_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_2")
    fullyConnectedLayer(16,"Name","fc_8")
    reluLayer("Name","relu_18")
    fullyConnectedLayer(64,"Name","fc_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalMaxPooling2dLayer("Name","gmpool_2")
    fullyConnectedLayer(16,"Name","fc_10")
    reluLayer("Name","relu_19")
    fullyConnectedLayer(64,"Name","fc_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_7")
    sigmoidLayer("Name","sigmoid_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)mean(X,3),"Name","avgpool2d_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)max(X,[],3),"Name","maxpool_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_2")
    convolution2dLayer([3 3],1,"Name","conv_17","Padding","same")
    sigmoidLayer("Name","sigmoid_6_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
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
    reluLayer("Name","relu_7_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_3")
    fullyConnectedLayer(16,"Name","fc_12")
    reluLayer("Name","relu_20")
    fullyConnectedLayer(64,"Name","fc_13")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalMaxPooling2dLayer("Name","gmpool_3")
    fullyConnectedLayer(16,"Name","fc_14")
    reluLayer("Name","relu_21")
    fullyConnectedLayer(64,"Name","fc_15")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_8")
    sigmoidLayer("Name","sigmoid_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)mean(X,3),"Name","avgpool2d_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)max(X,[],3),"Name","maxpool_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_3")
    convolution2dLayer([3 3],1,"Name","conv_18","Padding","same")
    sigmoidLayer("Name","sigmoid_6_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
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
    reluLayer("Name","relu_7_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_4")
    fullyConnectedLayer(16,"Name","fc_16")
    reluLayer("Name","relu_22")
    fullyConnectedLayer(64,"Name","fc_17")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalMaxPooling2dLayer("Name","gmpool_4")
    fullyConnectedLayer(16,"Name","fc_18")
    reluLayer("Name","relu_23")
    fullyConnectedLayer(64,"Name","fc_19")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_9")
    sigmoidLayer("Name","sigmoid_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)mean(X,3),"Name","avgpool2d_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)max(X,[],3),"Name","maxpool_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_4")
    convolution2dLayer([3 3],1,"Name","conv_21","Padding","same")
    sigmoidLayer("Name","sigmoid_6_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_5")
    convolution2dLayer([3 3],1,"Name","conv_19","Padding","same")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

lgraph = connectLayers(lgraph,"relu_1","conv_2");
lgraph = connectLayers(lgraph,"relu_1","addition_1/in1");
lgraph = connectLayers(lgraph,"relu_7","gapool");
lgraph = connectLayers(lgraph,"relu_7","gmpool");
lgraph = connectLayers(lgraph,"relu_7","multiplication/in2");
lgraph = connectLayers(lgraph,"fc_1","addition/in2");
lgraph = connectLayers(lgraph,"fc_3","addition/in1");
lgraph = connectLayers(lgraph,"sigmoid","multiplication/in1");
lgraph = connectLayers(lgraph,"multiplication","avgpool2d");
lgraph = connectLayers(lgraph,"multiplication","maxpool");
lgraph = connectLayers(lgraph,"multiplication","multiplication_6/in2");
lgraph = connectLayers(lgraph,"avgpool2d","depthcat/in2");
lgraph = connectLayers(lgraph,"maxpool","depthcat/in1");
lgraph = connectLayers(lgraph,"sigmoid_6","multiplication_6/in1");
lgraph = connectLayers(lgraph,"multiplication_6","addition_1/in2");
lgraph = connectLayers(lgraph,"addition_1","conv_8");
lgraph = connectLayers(lgraph,"addition_1","addition_2/in1");
lgraph = connectLayers(lgraph,"relu_7_1","gapool_1");
lgraph = connectLayers(lgraph,"relu_7_1","gmpool_1");
lgraph = connectLayers(lgraph,"relu_7_1","multiplication_1/in2");
lgraph = connectLayers(lgraph,"fc_5","addition_6/in2");
lgraph = connectLayers(lgraph,"fc_7","addition_6/in1");
lgraph = connectLayers(lgraph,"sigmoid_1","multiplication_1/in1");
lgraph = connectLayers(lgraph,"multiplication_1","avgpool2d_1");
lgraph = connectLayers(lgraph,"multiplication_1","maxpool_1");
lgraph = connectLayers(lgraph,"multiplication_1","multiplication_6_1/in2");
lgraph = connectLayers(lgraph,"avgpool2d_1","depthcat_1/in2");
lgraph = connectLayers(lgraph,"maxpool_1","depthcat_1/in1");
lgraph = connectLayers(lgraph,"sigmoid_6_1","multiplication_6_1/in1");
lgraph = connectLayers(lgraph,"multiplication_6_1","addition_2/in2");
lgraph = connectLayers(lgraph,"addition_2","conv_14");
lgraph = connectLayers(lgraph,"addition_2","addition_3/in1");
lgraph = connectLayers(lgraph,"relu_7_2","gapool_2");
lgraph = connectLayers(lgraph,"relu_7_2","gmpool_2");
lgraph = connectLayers(lgraph,"relu_7_2","multiplication_2/in2");
lgraph = connectLayers(lgraph,"fc_9","addition_7/in2");
lgraph = connectLayers(lgraph,"fc_11","addition_7/in1");
lgraph = connectLayers(lgraph,"sigmoid_2","multiplication_2/in1");
lgraph = connectLayers(lgraph,"multiplication_2","avgpool2d_2");
lgraph = connectLayers(lgraph,"multiplication_2","maxpool_2");
lgraph = connectLayers(lgraph,"multiplication_2","multiplication_6_2/in2");
lgraph = connectLayers(lgraph,"avgpool2d_2","depthcat_2/in2");
lgraph = connectLayers(lgraph,"maxpool_2","depthcat_2/in1");
lgraph = connectLayers(lgraph,"sigmoid_6_2","multiplication_6_2/in1");
lgraph = connectLayers(lgraph,"multiplication_6_2","addition_3/in2");
lgraph = connectLayers(lgraph,"addition_3","conv_16_5");
lgraph = connectLayers(lgraph,"addition_3","addition_4/in1");
lgraph = connectLayers(lgraph,"relu_7_3","gapool_3");
lgraph = connectLayers(lgraph,"relu_7_3","gmpool_3");
lgraph = connectLayers(lgraph,"relu_7_3","multiplication_3/in2");
lgraph = connectLayers(lgraph,"fc_13","addition_8/in2");
lgraph = connectLayers(lgraph,"fc_15","addition_8/in1");
lgraph = connectLayers(lgraph,"sigmoid_3","multiplication_3/in1");
lgraph = connectLayers(lgraph,"multiplication_3","avgpool2d_3");
lgraph = connectLayers(lgraph,"multiplication_3","maxpool_3");
lgraph = connectLayers(lgraph,"multiplication_3","multiplication_6_3/in2");
lgraph = connectLayers(lgraph,"avgpool2d_3","depthcat_3/in2");
lgraph = connectLayers(lgraph,"maxpool_3","depthcat_3/in1");
lgraph = connectLayers(lgraph,"sigmoid_6_3","multiplication_6_3/in1");
lgraph = connectLayers(lgraph,"multiplication_6_3","addition_4/in2");
lgraph = connectLayers(lgraph,"addition_4","conv_16_11");
lgraph = connectLayers(lgraph,"addition_4","addition_5/in1");
lgraph = connectLayers(lgraph,"relu_7_4","gapool_4");
lgraph = connectLayers(lgraph,"relu_7_4","gmpool_4");
lgraph = connectLayers(lgraph,"relu_7_4","multiplication_4/in2");
lgraph = connectLayers(lgraph,"fc_17","addition_9/in2");
lgraph = connectLayers(lgraph,"fc_19","addition_9/in1");
lgraph = connectLayers(lgraph,"sigmoid_4","multiplication_4/in1");
lgraph = connectLayers(lgraph,"multiplication_4","avgpool2d_4");
lgraph = connectLayers(lgraph,"multiplication_4","maxpool_4");
lgraph = connectLayers(lgraph,"multiplication_4","multiplication_6_4/in2");
lgraph = connectLayers(lgraph,"avgpool2d_4","depthcat_4/in2");
lgraph = connectLayers(lgraph,"maxpool_4","depthcat_4/in1");
lgraph = connectLayers(lgraph,"sigmoid_6_4","multiplication_6_4/in1");
lgraph = connectLayers(lgraph,"multiplication_6_4","addition_5/in2");

layers=lgraph;
end