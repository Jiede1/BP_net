#BP神经网络,实现的是一个三层，输入到输出分别是4，5，1层的网络（未加偏置）
#正向传播算法
def forward(dataSet,classLabels,W1,W2):
    m=dataSet.shape[0]
    X=np.hstack((np.ones((m,1)),dataSet))   #对数据集加偏置
    input_yin=W1.dot(np.hstack((np.ones((m,1)),dataSet)).T)
    output_yin=sigmoid(input_yin)
    output_yin_pianzhi=np.vstack((np.ones((1,m)),output_yin))  #对隐含层输出加偏置
    input_out=W2.dot(output_yin_pianzhi)
    output_out=sigmoid(input_out)
    predict=output_out
    return output_yin,predict  #返回隐含层和输出层的输出
def predict_end(output_out):    #最终预测结果
    predict=[]
    for i in range(output_out.shape[1]):
        if output_out[0][i]>=0.5:
            predict.append(1)
        else:
            predict.append(0)
    predict=np.array(predict).reshape(output_out.shape[1])
    return predict
def sigmoid(X):
    return 1/(1+exp(-X))
#反向传播算法    
def backward(dataSet,classLabels,output_yin,predict,W1,W2):
    delta_output=np.multiply(predict*(1-predict),classLabels-predict)  #输出层误差项
    W2_wupianzhi=W2[:,1:]  #去掉偏置项对应的权重项
    W1_wupianzhi=W1[:,1:]
    delta_yinhan=np.multiply(W2_wupianzhi.T*delta_output,output_yin*(1-output_yin))  #隐含层误差项
    return delta_output,delta_yinhan
def updatew(W1,W2,dataSet,delta_output,delta_yinhan,output_yin,alpha):
    #print(delta_output.shape,delta_yinhan.shape,output_yin.shape)
    W2[:,1:]=W2[:,1:]+alpha*delta_output.dot(output_yin.T)   #输出层权重更新
    W1[:,1:]=W1[:,1:]+alpha*delta_yinhan.dot(dataSet)        #隐含层权重更新
    W2[:,0]=W2[:,0]+alpha*delta_output.dot(np.ones((dataSet.shape[0],1)))   #输出层偏置项更新
    W1[:,:1]=W1[:,:1]+alpha*delta_yinhan.dot(np.ones((dataSet.shape[0],1)))   
    return W2,W1
#计算错误率，这里为了简单起见，只考虑输出是个标量的情况
def error_rate(classLabels,predict):
    rate=0
    for i in range(len(classLabels)):
        if classLabels[i]!=predict[i]:
            rate+=1
    return float(rate)/len(predict)

#假设输入的是从左到右是4，5，1层的网络（二类分类问题）
def read_iris():
    from sklearn.datasets import load_iris
    from sklearn import preprocessing
    data_set = load_iris()
    data_x = data_set.data 
    label = data_set.target 
    #preprocessing.scale(data_x, axis=0, with_mean=True, with_std=True, copy=False) 
    data_x=data[np.nonzero(label!=2)[0]]
    label=label[np.nonzero(label!=2)[0]]
    arr = np.arange(data_x.shape[0])
    np.random.shuffle(arr)   #打乱数据
    data_x=data_x[arr]
    label=label[arr]
    return data_x,label 
dataSet,classLabels=read_iris()
maxiter=1000
alpha=0.001
W1=np.hstack((np.random.random((5,1)),np.random.random((5,4))))-np.random.random(W1.shape) #加上偏置项
W2=np.hstack((np.random.random((1,1)),np.random.random((1,5))))-np.random.random(W2.shape)
for i in range(maxiter):
    output_yin,predict=forward(dataSet,classLabels,W1,W2)
    delta_output,delta_yinhan=backward(dataSet,classLabels,output_yin,predict,W1,W2)
    W2,W1=updatew(W1,W2,dataSet,delta_output,delta_yinhan,output_yin,alpha)
predict=predict_end(predict)
print(error_rate(classLabels,predict))
