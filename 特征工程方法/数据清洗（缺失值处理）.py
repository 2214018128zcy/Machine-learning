import pandas as pd

#创建包含缺失值的示例数据集
data={'A':[1,2,None,4],'B':[6,None,8,9]}
df=pd.DataFrame(data)

#删除包含缺失值的行
#df.dropna(inplace=True)



#使用均值填充缺失值
#df.fillna(df.mean(),inplace=True)


#使用插值方法填充缺失值
df.interpolate(method='linear',inplace=True)
print(df)

