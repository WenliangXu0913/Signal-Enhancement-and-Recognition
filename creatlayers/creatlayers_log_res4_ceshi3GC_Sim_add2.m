function [layers] = creatlayers_log_res4_ceshi3GC_Sim_add2
lgraph = layerGraph();

tempLayers = imageInputLayer([65 65 1],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)reshape(X,4225,64,[]),"Name","reshapeHWC_15_1_1_4_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1,"Name","conv_16_28_1_1_4_1","Padding","same")
    functionLayer(@(X)pagetranspose(reshape(X,4225,1,[])),"Name","reshapeCHW_14_1_1_4_1")
    softmaxLayer("Name","softmax_14_1_1_4_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    functionLayer(@(X1,X2)pagemtimes(X1,X2),"Name","matrixtimes_15_1_1_4_1")
    functionLayer(@(X)reshape(X,1,1,64,[]),"Name","reshapeSSCB_1_4_1")
    convolution2dLayer([1 1],16,"Name","conv_16_18_1","Padding","same")
    layerNormalizationLayer("Name","layernorm_1_4_1")
    reluLayer("Name","relu_2_4_1")
    convolution2dLayer([1 1],64,"Name","conv_18_14_1_1_4_1","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    functionLayer(@(X1,X2)X1+X2,"Name","broadcast_addition_1_4_1")
    convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1_14_1_1")
    convolution2dLayer([3 3],64,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_1_1")
    convolution2dLayer([3 3],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_4")
    convolution2dLayer([3 3],64,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_1_2")
    convolution2dLayer([3 3],64,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_6")
    convolution2dLayer([3 3],64,"Name","conv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_1_3")
    convolution2dLayer([3 3],64,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_8")
    convolution2dLayer([3 3],64,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_1_4")
    convolution2dLayer([3 3],64,"Name","conv_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")
    reluLayer("Name","relu_10")
    convolution2dLayer([3 3],64,"Name","conv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")
    reluLayer("Name","relu_1_5")
    convolution2dLayer([3 3],64,"Name","conv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")
    reluLayer("Name","relu_12")
    convolution2dLayer([3 3],64,"Name","conv_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_12")
    reluLayer("Name","relu_1_6")
    convolution2dLayer([3 3],64,"Name","conv_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_13")
    reluLayer("Name","relu_14")
    convolution2dLayer([3 3],64,"Name","conv_15","Padding","same")
    batchNormalizationLayer("Name","batchnorm_14")
    reluLayer("Name","relu_1_7")
    convolution2dLayer([3 3],64,"Name","conv_16_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_1")
    reluLayer("Name","relu_16_1")
    convolution2dLayer([3 3],64,"Name","conv_16_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_2")
    reluLayer("Name","relu_1_8")
    convolution2dLayer([3 3],64,"Name","conv_16_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_3")
    reluLayer("Name","relu_16_3")
    convolution2dLayer([3 3],64,"Name","conv_16_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_4")
    reluLayer("Name","relu_1_9")
    convolution2dLayer([3 3],64,"Name","conv_16_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_5")
    reluLayer("Name","relu_16_5")
    convolution2dLayer([3 3],64,"Name","conv_16_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_6")
    reluLayer("Name","relu_1_10")
    convolution2dLayer([3 3],64,"Name","conv_16_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_7")
    reluLayer("Name","relu_16_7")
    convolution2dLayer([3 3],64,"Name","conv_16_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_8")
    reluLayer("Name","relu_1_11")
    convolution2dLayer([3 3],64,"Name","conv_16_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_9")
    reluLayer("Name","relu_16_9")
    convolution2dLayer([3 3],64,"Name","conv_16_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_10")
    reluLayer("Name","relu_1_12")
    convolution2dLayer([3 3],64,"Name","conv_16_5_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_5_1")
    reluLayer("Name","relu_16_5_1")
    convolution2dLayer([3 3],64,"Name","conv_16_6_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_6_1")
    reluLayer("Name","relu_1_10_1")
    convolution2dLayer([3 3],64,"Name","conv_16_7_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_7_1")
    reluLayer("Name","relu_16_7_1")
    convolution2dLayer([3 3],64,"Name","conv_16_8_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_8_1")
    reluLayer("Name","relu_1_11_1")
    convolution2dLayer([3 3],64,"Name","conv_16_9_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_9_1")
    reluLayer("Name","relu_16_9_1")
    convolution2dLayer([3 3],64,"Name","conv_16_8_1_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_8_1_1")
    reluLayer("Name","relu_1_11_1_1")
    convolution2dLayer([3 3],64,"Name","conv_16_9_1_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_9_1_1")
    reluLayer("Name","relu_16_9_1_1")
    convolution2dLayer([3 3],64,"Name","conv_16_10_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15_10_1")
    reluLayer("Name","relu_1_12_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = functionLayer(@(X)reshape(X,4225,64,[]),"Name","reshapeHWC_15_1_1_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1,"Name","conv_16_28_1_1_4","Padding","same")
    functionLayer(@(X)pagetranspose(reshape(X,4225,1,[])),"Name","reshapeCHW_14_1_1_4")
    softmaxLayer("Name","softmax_14_1_1_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    functionLayer(@(X1,X2)pagemtimes(X1,X2),"Name","matrixtimes_15_1_1_4")
    functionLayer(@(X)reshape(X,1,1,64,[]),"Name","reshapeSSCB_1_4")
    convolution2dLayer([1 1],16,"Name","conv_16_18","Padding","same")
    layerNormalizationLayer("Name","layernorm_1_4")
    reluLayer("Name","relu_2_4")
    convolution2dLayer([1 1],64,"Name","conv_18_14_1_1_4","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    functionLayer(@(X1,X2)X1+X2,"Name","broadcast_addition_1_4")
    convolution2dLayer([3 3],1,"Name","conv_19","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_4")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

lgraph = connectLayers(lgraph,"imageinput","conv_1");
lgraph = connectLayers(lgraph,"imageinput","addition_4/in1");
lgraph = connectLayers(lgraph,"relu_1","reshapeHWC_15_1_1_4_1");
lgraph = connectLayers(lgraph,"relu_1","conv_16_28_1_1_4_1");
lgraph = connectLayers(lgraph,"relu_1","broadcast_addition_1_4_1/in1");
lgraph = connectLayers(lgraph,"reshapeHWC_15_1_1_4_1","matrixtimes_15_1_1_4_1/in2");
lgraph = connectLayers(lgraph,"softmax_14_1_1_4_1","matrixtimes_15_1_1_4_1/in1");
lgraph = connectLayers(lgraph,"conv_18_14_1_1_4_1","broadcast_addition_1_4_1/in2");
lgraph = connectLayers(lgraph,"relu_1_12_1","reshapeHWC_15_1_1_4");
lgraph = connectLayers(lgraph,"relu_1_12_1","conv_16_28_1_1_4");
lgraph = connectLayers(lgraph,"relu_1_12_1","broadcast_addition_1_4/in1");
lgraph = connectLayers(lgraph,"reshapeHWC_15_1_1_4","matrixtimes_15_1_1_4/in2");
lgraph = connectLayers(lgraph,"softmax_14_1_1_4","matrixtimes_15_1_1_4/in1");
lgraph = connectLayers(lgraph,"conv_18_14_1_1_4","broadcast_addition_1_4/in2");
lgraph = connectLayers(lgraph,"conv_19","addition_4/in2");
layers = lgraph;
end