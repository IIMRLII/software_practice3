[TOC]

# 实验一

![image-20220414143224967](README.assets/image-20220414143224967.png)

测试AndroidStudio

![image-20220414150703145](README.assets/image-20220414150703145.png)

安装Anaconda

![image-20220414153303176](README.assets/image-20220414153303176.png)

调试jupyter

![image-20220414153725652](README.assets/image-20220414153725652.png)

# 实验二

先将PPT中功能、布局代码实现

![image-20220421224715184](README.assets/image-20220421224715184.png)

![image-20220421224733961](README.assets/image-20220421224733961.png)

## 扩展功能（注释中为bundle实现）

在导航视图中设置参数并build，生成参数文件

![image-20220429193631321](README.assets/image-20220429193631321.png)

![image-20220429193644244](README.assets/image-20220429193644244.png)

在第一个页面中使用safeargs进行数据传输

![image-20220429193655439](README.assets/image-20220429193655439.png)

设置第二个页面的代码，接收并显示数据，设置随机数

![image-20220429193705941](README.assets/image-20220429193705941.png)

效果展示

![image-20220421224848479](README.assets/image-20220421224848479.png)

![image-20220421224856520](README.assets/image-20220421224856520.png)

# 实验三

先按照CSDN中步骤下载并build相应文件

![image-20220505151910909](README.assets/image-20220505151910909.png)

添加对应识别代码

![image-20220505151940107](README.assets/image-20220505151940107.png)

安装到真机

<img src="README.assets/image-20220505152213577.png" alt="image-20220505152213577" style="zoom: 33%;" />

给予权限

<img src="README.assets/image-20220505152227198.png" alt="image-20220505152227198" style="zoom:33%;" />

识别对应花卉

<img src="README.assets/image-20220505152238123.png" alt="image-20220505152238123" style="zoom:33%;" />

<img src="README.assets/image-20220505152247059.png" alt="image-20220505152247059" style="zoom:33%;" />

<img src="README.assets/image-20220505152254557.png" alt="image-20220505152254557" style="zoom:33%;" />

# 实验四

## 选择排序


```python
def selection_sort(arr):
    if len(arr) < 0:
        return
    for i in range(len(arr)):
        min = i
        for j in range(i, len(arr)):
            if(arr[j] < arr[min]):
                min = j
        temp = arr[i]
        arr[i] = arr[min]
        arr[min] = temp
```


```python
def test():
    arr = [1, 3, 6, 22, 3, 5, 2, 3, 5, 2, 35, -2]
    selection_sort(arr)
    print(arr)
```


```python
test();
```

    [-2, 1, 2, 2, 3, 3, 3, 5, 5, 6, 22, 35]


## 设置


```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### 读取数据集 


```python
df = pd.read_csv('fortune500.csv')
```

## 检查数据集


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25495</th>
      <td>2005</td>
      <td>496</td>
      <td>Wm. Wrigley Jr.</td>
      <td>3648.6</td>
      <td>493</td>
    </tr>
    <tr>
      <th>25496</th>
      <td>2005</td>
      <td>497</td>
      <td>Peabody Energy</td>
      <td>3631.6</td>
      <td>175.4</td>
    </tr>
    <tr>
      <th>25497</th>
      <td>2005</td>
      <td>498</td>
      <td>Wendy's International</td>
      <td>3630.4</td>
      <td>57.8</td>
    </tr>
    <tr>
      <th>25498</th>
      <td>2005</td>
      <td>499</td>
      <td>Kindred Healthcare</td>
      <td>3616.6</td>
      <td>70.6</td>
    </tr>
    <tr>
      <th>25499</th>
      <td>2005</td>
      <td>500</td>
      <td>Cincinnati Financial</td>
      <td>3614.0</td>
      <td>584</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.columns = ['year', 'rank', 'company', 'revenue', 'profit']
```


```python
len(df)
```




    25500




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit      object
    dtype: object




```python
non_numberic_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numberic_profits].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>228</th>
      <td>1955</td>
      <td>229</td>
      <td>Norton</td>
      <td>135.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>290</th>
      <td>1955</td>
      <td>291</td>
      <td>Schlitz Brewing</td>
      <td>100.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>294</th>
      <td>1955</td>
      <td>295</td>
      <td>Pacific Vegetable Oil</td>
      <td>97.9</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>296</th>
      <td>1955</td>
      <td>297</td>
      <td>Liebmann Breweries</td>
      <td>96.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>352</th>
      <td>1955</td>
      <td>353</td>
      <td>Minneapolis-Moline</td>
      <td>77.4</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>

</div>




```python
len(df.profit[non_numberic_profits])
```




    369




```python
bin_sizes, _, _ = plt.hist(df.year[non_numberic_profits], bins=range(1955, 2006))
```


​    
![png](README.assets/output_16_0.png)
​    



```python
df = df.loc[~non_numberic_profits]
df.profit = df.profit.apply(pd.to_numeric)
```


```python
len(df)
```




    25131




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit     float64
    dtype: object



## 使用matplotlib进行绘图


```python
group_by_year = df.loc[:, ['year', 'revenue', 'profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x=0, y=0)
```


```python
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1955 to 2005', 'Profit (millions)')
```


​    
![png](README.assets/output_22_0.png)
​    



```python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955 to 2005', 'Revenue (millions)')
```


​    
![png](README.assets/output_23_0.png)
​    



```python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha=0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots(ncols=2)
title = 'Increase in mean and std Fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'revenues', 'Revenue (millions)')
fig.set_size_inches(14, 4)
fig.tight_layout()
```


​    
![png](README.assets/output_24_0.png)
​    

