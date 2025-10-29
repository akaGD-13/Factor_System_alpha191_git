import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


# def alpha21(data):
#     '''
#     通过输入的数据集生成一个特定因子,  逻辑是用过去6日的6日均值序列(实际上使用了11日数据),
#     对[1,2,3,4,5,6]这个序列做回归, 对它做回归的含义是, 以[1,2,3,4,5,6]作为自变量.
#     最终因子为这个回归的回归系数.
#     本因子对线性回归的计算基于sklearn, 计算时间大约需要50分钟

#     Parameters
#     ----------
#     data : DataFrame
#         输入一个已经经过计算的数据集, 本因子至少需要输入close

#     Returns
#     -------
#     输出一个各交易日收盘后可计算得到的因子值面板
#     备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

#     '''
#     model = LinearRegression()
#     X = np.arange(1, 7).reshape(-1, 1)

#     close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')

#     factor1 = close_panel.rolling(6).mean()

#     df = pd.DataFrame(index=factor1.index, columns=factor1.columns)

#     for row in tqdm(range(5, close_panel.shape[0])):
#         for col in range(close_panel.shape[1]):
#             # 需要至少6个数据点
#             Y = close_panel.iloc[row - 5:row + 1, col].values.reshape(-1, 1)  # 目标变量
#             if np.isnan(Y).sum() != 0:
#                 continue
#             model.fit(X, Y)  # 拟合回归模型
#             df.iloc[row, col] = model.coef_[0, 0]  # 取回归系数（斜率）

#     return df


# def alpha24(data):
#     '''
#     通过输入的数据集生成一个特定因子, 构建逻辑如下: 
#         因子核心为close-delay(close, 5)
#         我们以(5,1)的参数计算这个因子核心的SMA. 
#         计算时间预计需要30分钟. 
        
#     Parameters
#     ----------
#     data : DataFrame
#         输入一个已经经过计算的数据集, 本因子至少需要输入close

#     Returns
#     -------
#     输出一个各交易日收盘后可计算得到的因子值面板
#     备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

#     '''
#     close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')
    
#     factor = close_panel - close_panel.shift(5)

#     df = pd.DataFrame(index=factor.index, columns=factor.columns)
    
#     for row in tqdm(range(1, factor.shape[0])):
#         for col in range(factor.shape[1]):
#             Y_i = df.iloc[row-1, col]
#             A = factor.iloc[row, col]
#             if not np.isnan(A) and np.isnan(Y_i):
#                 Y = A
#             elif np.isnan(A):
#                 Y = np.nan
#             else:
#                 Y = A / 5 + 4 * Y_i / 5
#             df.iloc[row, col] = Y
    
#     return df


# def alpha27(data):
#     '''
#     通过输入的数据集生成一个特定因子, 因子核心包括两部分: 
#         1. close/delay(close, 3) - 1
#         2. close / delay(close, 6) - 1
#     将3日涨跌幅和6日涨跌幅相加*100, 然后计算其参数为12的WMA
#     WMA的构建方式是, 对序列以0.9的指数衰减加权.
        
        
#     Parameters
#     ----------
#     data : DataFrame
#         输入一个已经经过计算的数据集, 本因子至少需要输入close

#     Returns
#     -------
#     输出一个各交易日收盘后可计算得到的因子值面板
#     备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

#     '''
#     close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')
#     factor1 = close_panel / close_panel.shift(3) - 1
#     factor2 = close_panel / close_panel.shift(6) - 1
#     factor = (factor1 + factor2) * 100
    
#     weight = []
#     for i in range(12, 0, -1):
#         weight.append(0.9 ** i)
#     weight = np.array(weight) / sum(weight)
    
#     df = factor.rolling(window=12).apply(lambda x: np.sum(x * weight[-len(x):]), raw=True)
    
#     return df


# def alpha28(data):
#     '''
#     通过输入的数据集生成一个特定因子, 因子核心为: 
#         收盘价与九日最低价的最小值之差与九日最高价的最大值和九日最低价的最小值之差的比
#         对这一核心*100, 计算(3,1)的SMA
#         factor1为(3,1)的SMA
#         factor2为factor1参数(3,1)的SMA
#         最终因子为3*factor1 - 2*factor2
        
#     因子计算预计需要一个小时30分钟. 
        
        
#     Parameters
#     ----------
#     data : DataFrame
#         输入一个已经经过计算的数据集, 本因子至少需要输入close, high, low
#     Returns
#     -------
#     输出一个各交易日收盘后可计算得到的因子值面板
#     备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理
            
#     '''
#     close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')
#     high_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='high')
#     low_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='low')
    
#     factor = (close_panel - low_panel.rolling(9).min()) / (high_panel.rolling(9).max() - low_panel.rolling(9).min()) * 100
    
#     factor1 = pd.DataFrame(index=factor.index, columns=factor.columns)
    
#     for row in tqdm(range(1, factor.shape[0])):
#         for col in range(factor.shape[1]):
#             Y_i = factor1.iloc[row-1, col]
#             A = factor.iloc[row, col]
#             if not np.isnan(A) and np.isnan(Y_i):
#                 Y = A
#             elif np.isnan(A):
#                 Y = np.nan
#             else:
#                 Y = A / 3 + 2 * Y_i / 3
#             factor1.iloc[row, col] = Y
            
#     factor2 = pd.DataFrame(index=factor.index, columns=factor.columns)  
      
#     for row in tqdm(range(1, factor1.shape[0])):
#         for col in range(factor1.shape[1]):
#             Y_i = factor2.iloc[row-1, col]
#             A = factor1.iloc[row, col]
#             if not np.isnan(A) and np.isnan(Y_i):
#                 Y = A
#             elif np.isnan(A):
#                 Y = np.nan
#             else:
#                 Y = A / 3 + 2 * Y_i / 3
#             factor2.iloc[row, col] = Y
            
#     df = 3 * factor1 - 2 * factor2
    
#     return df


def alpha21(data: pd.DataFrame) -> pd.DataFrame:
    '''
    通过输入的数据集生成一个特定因子,  逻辑是用过去6日的6日均值序列(实际上使用了11日数据),
    对[1,2,3,4,5,6]这个序列做回归, 对它做回归的含义是, 以[1,2,3,4,5,6]作为自变量.
    最终因子为这个回归的回归系数.
    本因子对线性回归的计算基于sklearn, 计算时间大约需要50分钟

    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入close

    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

    '''
    model = LinearRegression()
    X = np.arange(1, 7).reshape(-1, 1)  # 自变量序列 [1,2,3,4,5,6]

    close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')

    # 计算6日均线面板
    factor1 = close_panel.rolling(6).mean()

    # 创建空结果DataFrame
    df = pd.DataFrame(index=factor1.index, columns=factor1.columns)

    # 修改1: 循环起始点改为10，因为需要11天数据(6日均线的6个连续值需要11天)
    # 修改2: 使用均线序列(factor1)而不是原始收盘价(close_panel)
    for row in tqdm(range(10, close_panel.shape[0])):
        for col in range(close_panel.shape[1]):
            # 修改3: 取6日均线序列的连续6个值 (当前行往前推5天到当前行)
            # 注意: iloc切片是左闭右开，所以取[row-5:row+1]实际取6行
            Y = factor1.iloc[row - 5:row + 1, col].values.reshape(-1, 1)  # 使用均线值

            # 检查NaN值
            if np.isnan(Y).any():
                continue

            model.fit(X, Y)  # 拟合回归模型
            # 修改4: 存储回归系数（斜率）
            df.iloc[row, col] = model.coef_[0, 0]

    return df.stack().reset_index().rename(columns={0: 'alpha21'})[['trade_date', 'ts_code', 'alpha21']]

def alpha22(data: pd.DataFrame) -> pd.DataFrame:
    '''
    通过输入的数据集生成一个特定因子, 构建逻辑如下:
        首先构建因子核心, 组成为close/mean(close, 6) - 1
        我们把这个核心称为K, 那么因子值为K/12 + 11*F(i-1)/12
        其中, F(i-1)是上一期因子. 我们将第一期因子的初始值, 直接确定为K本身
        原文中公式疑似有误, 经过核对, 我们合理怀疑其公式为SMA, 而非SMEAN, 因此按这种方法计算.
        因子计算耗时预计30分钟.

    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入close

    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

    '''
    close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')

    # 计算因子核心K: close/6日均值 - 1
    factor = close_panel / close_panel.rolling(6).mean() - 1

    df = pd.DataFrame(index=factor.index, columns=factor.columns)

    # 修改1: 初始化第0行（第一期）为K值本身
    # 根据注释要求："我们将第一期因子的初始值, 直接确定为K本身"
    df.iloc[0] = factor.iloc[0]

    # 修改2: 从第1行开始计算（而不是第0行）
    for row in tqdm(range(1, factor.shape[0])):
        for col in range(factor.shape[1]):
            Y_i = df.iloc[row - 1, col]  # 上一期因子值
            A = factor.iloc[row, col]  # 当前核心K值

            # 处理NaN值
            if np.isnan(A):
                Y = np.nan
            elif np.isnan(Y_i):
                # 修改3: 当上一期因子为NaN时，使用当前K值作为初始值
                # 符合"第一期因子的初始值直接确定为K本身"的逻辑
                Y = A
            else:
                # SMA公式: F(t) = K(t)/12 + 11*F(t-1)/12
                Y = A / 12 + 11 * Y_i / 12

            df.iloc[row, col] = Y

    return df.stack().reset_index().rename(columns={0: 'alpha22'})[['trade_date', 'ts_code', 'alpha22']]

def alpha23(data: pd.DataFrame) -> pd.DataFrame:
    '''
    通过输入的数据集生成一个特定因子, 因子核心包括两部分:
        1. 当收盘价上涨时, 使用20日标准差
        2. 当收盘价下跌时, 使用20日标准差
        分别对这两部分计算20日SMA, 然后计算第一部分在两部分之和中的占比*100
        因子计算预计需要一小时30分钟.

    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入close

    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

    '''
    # 1. 创建收盘价面板
    close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')

    # 2. 计算20日标准差面板
    std_panel = close_panel.rolling(20).std()

    # 3. 计算收盘价变化 - 修改点：移除了periods参数
    close_diff = close_panel.diff()  # 修改：移除了periods=1参数

    # 4. 创建涨跌信号 - 重大修改点：完全重构了信号处理逻辑
    sign = np.where(close_diff > 0, True, np.where(close_diff <= 0, False, np.nan))

    # 5. 构建因子核心 - 重大修改点：重构了因子计算逻辑
    factor1 = pd.DataFrame(np.where(sign == 1, std_panel, np.where(sign == 0, 0, np.nan)),
                           index=close_diff.index, columns=close_diff.columns)
    factor2 = pd.DataFrame(np.where(sign == 0, std_panel, np.where(sign == 1, 0, np.nan)),
                           index=close_diff.index, columns=close_diff.columns)

    # 6. 创建SMA面板 - 修改点：简化了初始化方式
    sma1 = pd.DataFrame(columns=factor1.columns, index=factor1.index)
    sma2 = pd.DataFrame(columns=factor2.columns, index=factor2.index)

    # 7. SMA计算循环 - 修改点：重构了条件判断逻辑
    for row in tqdm(range(1, factor1.shape[0])):  # 修改：使用shape[0]而不是len(index)
        for col in range(factor1.shape[1]):  # 修改：使用shape[1]而不是len(columns)
            # 修改：使用iloc而不是iat
            Y1_i = sma1.iloc[row - 1, col]
            Y2_i = sma2.iloc[row - 1, col]
            A1 = factor1.iloc[row, col]
            A2 = factor2.iloc[row, col]

            # 重构了条件判断逻辑：
            if not np.isnan(A1) and np.isnan(Y1_i):
                Y1 = A1
            elif np.isnan(A1):
                Y1 = np.nan
            else:
                Y1 = A1 / 20 + 19 * Y1_i / 20

            # 同样重构了因子2的条件判断
            if not np.isnan(A2) and np.isnan(Y2_i):
                Y2 = A2
            elif np.isnan(A2):
                Y2 = np.nan
            else:
                Y2 = A2 / 20 + 19 * Y2_i / 20

            # 存储计算结果 - 修改：使用iloc而不是iat
            sma1.iloc[row, col] = Y1
            sma2.iloc[row, col] = Y2

    # 9. 计算最终因子值 - 修改点：简化了计算方式，移除了错误处理
    df = sma1 / (sma1 + sma2) * 100

    return df.stack().reset_index().rename(columns={0: 'alpha23'})[['trade_date', 'ts_code', 'alpha23']]

def alpha24(data: pd.DataFrame) -> pd.DataFrame:
    '''
    通过输入的数据集生成一个特定因子, 构建逻辑如下:
        因子核心为close-delay(close, 5)
        我们以(5,1)的参数计算这个因子核心的SMA.
        计算时间预计需要30分钟.

    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入close

    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

    '''
    # 创建收盘价面板（日期为索引，股票为列）
    close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')

    # 计算因子核心：close - 5天前的close
    factor_core = close_panel - close_panel.shift(5)

    # 计算5日简单移动平均(SMA)
    df = factor_core.rolling(5).mean()

    return df.stack().reset_index().rename(columns={0: 'alpha24'})[['trade_date', 'ts_code', 'alpha24']]


# def alpha25(input: pd.DataFrame) -> pd.DataFrame:
#     data = input.sort_values(['ts_code', 'trade_date'])
#     close = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')
#     ret   = pd.pivot(data=data, columns='ts_code', index='trade_date', values='pct_chg')
#     vol   = pd.pivot(data=data, columns='ts_code', index='trade_date', values='volume')

#     # 1) 7-day close diff
#     factor1 = close - close.shift(7)

#     # 2) decay of (vol / mean(vol,20)) over 9 days
#     ratio = vol / vol.rolling(20, min_periods=10).mean()   # allow partial early
#     w = np.arange(9, 0, -1, dtype=float)
#     w /= w.sum()
#     factor2 = ratio.rolling(9, min_periods=5).apply(lambda x: np.sum(x * w[-len(x):]), raw=True)

#     # 3) 250-day sum of returns (requires 250d)
#     factor3 = ret.rolling(250, min_periods=250).sum()

#     # Cross-sectional ranks (only where available)
#     rank2 = factor2.rank(axis=1, method='min')
#     rank3 = factor3.rank(axis=1, method='min')

#     # Combine: only multiply on rows where all three exist
#     df = factor1 * (1 - rank2) * (1 + rank3)

#     out = (
#         df.stack(dropna=True)
#           .rename('alpha25')
#           .reset_index()[['trade_date','ts_code','alpha25']]
#     )
#     return out


def alpha26(data: pd.DataFrame) -> pd.DataFrame:
    '''
    通过输入的数据集生成一个特定因子, 构建逻辑如下:
        因子核心共有两个:
            1. 收盘价七日均值和收盘价的价差
            2. VWAP和5日前收盘价在230日序列的相关系数
        最终因子等于二者相加

    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入close和vwap

    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

    '''
    vwap_panel = pd.pivot(data=data, index='trade_date', columns='ts_code', values='vwap')
    close_panel = pd.pivot(data=data, index='trade_date', columns='ts_code', values='close')

    # 计算第一个因子: 7日收盘均值和当日收盘价的差值
    factor1 = close_panel.rolling(7).mean() - close_panel

    # 修正第二个因子的计算: 需要分别对每只股票计算VWAP和5日前收盘价的230日滚动相关系数
    factor2 = pd.DataFrame(index=vwap_panel.index, columns=vwap_panel.columns)

    # 循环计算每只股票的相关系数
    for ts_code in close_panel.columns:
        # 创建临时DataFrame包含需要的两列数据
        temp_df = pd.DataFrame({
            'vwap': vwap_panel[ts_code],
            'close_shifted': close_panel[ts_code].shift(5)  # 5日前收盘价
        })
        
        # 计算230日滚动相关系数
        factor2[ts_code] = temp_df['vwap'].rolling(230).corr(temp_df['close_shifted'])

    # 组合两个因子
    df = factor1 + factor2

    return df.stack().reset_index().rename(columns={0: 'alpha26'})[['trade_date', 'ts_code', 'alpha26']]

def alpha27(data: pd.DataFrame) -> pd.DataFrame:
    '''
    通过输入的数据集生成一个特定因子, 因子核心包括两部分:
        1. close/delay(close, 3) - 1
        2. close / delay(close, 6) - 1
    将3日涨跌幅和6日涨跌幅相加*100, 然后计算其参数为12的WMA
    WMA的构建方式是, 对序列以0.9的指数衰减加权.

    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入close

    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理
    '''
    close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')
    factor1 = close_panel / close_panel.shift(3) - 1
    factor2 = close_panel / close_panel.shift(6) - 1
    factor = (factor1 + factor2) * 100

    # 构建指数衰减权重基础数组（从远到近：0.9^12, 0.9^11, ... 0.9^1）
    weight_base = np.array([0.9 ** i for i in range(12, 0, -1)])

    # 定义WMA计算函数，动态处理不同窗口长度
    def wma_func(x):
        n = len(x)
        if n == 0:
            return np.nan
        # 取权重数组的最后n个权重（对应最近的n个数据点）
        weights = weight_base[-n:]
        # 关键修改：对当前窗口的权重进行归一化（确保权重和为1）
        normalized_weights = weights / weights.sum()
        return np.sum(x * normalized_weights)

    # 应用WMA计算（window=12）
    df = factor.rolling(window=12).apply(wma_func, raw=True)

    return df.stack().reset_index().rename(columns={0: 'alpha27'})[['trade_date', 'ts_code', 'alpha27']]

def alpha28(data: pd.DataFrame) -> pd.DataFrame:
    '''
    通过输入的数据集生成一个特定因子, 因子核心为:
        收盘价与九日最低价的最小值之差与九日最高价的最大值和九日最低价的最小值之差的比
        对这一核心*100, 计算(3,1)的SMA
        factor1为(3,1)的SMA
        factor2为factor1参数(3,1)的SMA
        最终因子为3*factor1 - 2*factor2

    因子计算预计需要一个小时30分钟.


    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入close, high, low
    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

    '''
    close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')
    high_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='high')
    low_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='low')

    # 计算核心因子：(收盘价 - 9日最低价) / (9日最高价 - 9日最低价) * 100
    factor = (close_panel - low_panel.rolling(9).min()) / (
                high_panel.rolling(9).max() - low_panel.rolling(9).min()) * 100

    # 计算SMA(简单移动平均)
    factor1 = factor.rolling(window=3).mean()
    factor2 = factor1.rolling(window=3).mean()

    # 最终因子：3*factor1 - 2*factor2
    df = 3 * factor1 - 2 * factor2

    return df.stack().reset_index().rename(columns={0: 'alpha28'})[['trade_date', 'ts_code', 'alpha28']]

def alpha29(data: pd.DataFrame) -> pd.DataFrame:
    '''
    通过输入的数据集生成一个特定因子, 因子为六日收益率和成交量的乘积
    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入vol和close

    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

    '''
    # 创建收盘价透视面板（日期为索引，股票代码为列）
    close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')

    # 创建成交量透视面板（日期为索引，股票代码为列）
    vol_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='volume')

    # 计算六日收益率：(当前收盘价/6天前收盘价 - 1)
    # 然后乘以当日成交量
    df = (close_panel / close_panel.shift(6) - 1) * vol_panel

    return df.stack().reset_index().rename(columns={0: 'alpha29'})[['trade_date', 'ts_code', 'alpha29']]


import pandas as pd
import numpy as np

def alpha77(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha77: MIN(RANK(DECAYLINEAR((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH), 20)),
                 RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(vol,40), 3), 6)))
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()

    # part 1
    df['temp1']  = ((df['high'] + df['low']) / 2 + df['high']) - (df['vwap'] + df['high'])
    df['decay1'] = df.groupby('ts_code')['temp1'].transform(lambda x: x.ewm(alpha=2/(20+1), adjust=False).mean())
    df['rank1']  = df.groupby('trade_date')['decay1'].rank()

    # part 2 (FIXED rolling corr)
    df['mean_volume'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=40, min_periods=1).mean())
    df['hl_avg']      = (df['high'] + df['low']) / 2
    df['corr'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['hl_avg'].rolling(window=3, min_periods=1).corr(g['mean_volume']))
          .reset_index(level=0, drop=True)
    )
    df['decay2'] = df.groupby('ts_code')['corr'].transform(lambda x: x.ewm(alpha=2/(6+1), adjust=False).mean())
    df['rank2']  = df.groupby('trade_date')['decay2'].rank()

    df['alpha77'] = df[['rank1', 'rank2']].min(axis=1)
    return df[['trade_date', 'ts_code', 'alpha77']]


def alpha83(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha83: (-1 * RANK(COVIANCE(RANK(HIGH), RANK(vol), 5)))
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['rank_high']   = df.groupby('trade_date')['high'].rank()
    df['rank_volume'] = df.groupby('trade_date')['volume'].rank()

    # FIXED rolling covariance
    df['cov'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['rank_high'].rolling(window=5, min_periods=1).cov(g['rank_volume']))
          .reset_index(level=0, drop=True)
    )
    df['rank_cov'] = df.groupby('trade_date')['cov'].rank()
    df['alpha83']  = -1 * df['rank_cov']
    return df[['trade_date', 'ts_code', 'alpha83']]


def alpha90(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha90: (RANK(CORR(RANK(VWAP), RANK(vol), 5)) * -1)
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['rank_vwap']   = df.groupby('trade_date')['vwap'].rank()
    df['rank_volume'] = df.groupby('trade_date')['volume'].rank()

    # FIXED rolling corr
    df['corr_rank'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['rank_vwap'].rolling(window=5, min_periods=1).corr(g['rank_volume']))
          .reset_index(level=0, drop=True)
    )
    df['rank_corr'] = df.groupby('trade_date')['corr_rank'].rank()
    df['alpha90']   = -1 * df['rank_corr']
    return df[['trade_date', 'ts_code', 'alpha90']]


def alpha91(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha91: ((RANK((CLOSE - MAX(CLOSE, 5))) * RANK(CORR((MEAN(vol,40)), LOW, 5))) * -1)
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['max_close_5']    = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(window=5, min_periods=1).max())
    df['close_diff']     = df['close'] - df['max_close_5']
    df['rank_close_diff']= df.groupby('trade_date')['close_diff'].rank()

    df['mean_volume_40'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=40, min_periods=1).mean())

    # FIXED rolling corr
    df['corr_volume_low'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['mean_volume_40'].rolling(window=5, min_periods=1).corr(g['low']))
          .reset_index(level=0, drop=True)
    )
    df['rank_corr_volume_low'] = df.groupby('trade_date')['corr_volume_low'].rank()

    df['alpha91'] = -1 * (df['rank_close_diff'] * df['rank_corr_volume_low'])
    return df[['trade_date', 'ts_code', 'alpha91']]


def alpha92(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha92: (MAX(RANK(DECAYLINEAR(DELTA((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)),
               TSRANK(DECAYLINEAR(ABS(CORR((MEAN(vol,180)), CLOSE, 13)), 5), 15)) * -1)
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()

    df['weighted_price']   = df['close'] * 0.35 + df['vwap'] * 0.65
    df['delta_weighted']   = df.groupby('ts_code')['weighted_price'].transform(lambda x: x.diff(2))
    df['decay_linear_3']   = df.groupby('ts_code')['delta_weighted'].transform(lambda x: x.ewm(alpha=2/(3+1), adjust=False).mean())
    df['rank_decay_3']     = df.groupby('trade_date')['decay_linear_3'].rank()

    df['mean_volume_180']  = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=180, min_periods=1).mean())

    # FIXED rolling corr
    df['corr_volume_close'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['mean_volume_180'].rolling(window=13, min_periods=1).corr(g['close']))
          .reset_index(level=0, drop=True)
    )
    df['abs_corr']        = df['corr_volume_close'].abs()
    df['decay_linear_4']  = df.groupby('ts_code')['abs_corr'].transform(lambda x: x.ewm(alpha=2/(5+1), adjust=False).mean())
    df['tsrank_decay_4']  = df.groupby('trade_date')['decay_linear_4'].rank(pct=True)

    df['max_rank'] = df[['rank_decay_3', 'tsrank_decay_4']].max(axis=1)
    df['alpha92']  = -1 * df['max_rank']
    return df[['trade_date', 'ts_code', 'alpha92']]


def alpha99(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha99: (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['rank_close']  = df.groupby('trade_date')['close'].rank()
    df['rank_volume'] = df.groupby('trade_date')['volume'].rank()

    # FIXED rolling covariance
    df['cov_rank'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['rank_close'].rolling(window=5, min_periods=1).cov(g['rank_volume']))
          .reset_index(level=0, drop=True)
    )
    df['rank_cov'] = df.groupby('trade_date')['cov_rank'].rank()
    df['alpha99']  = -1 * df['rank_cov']
    return df[['trade_date', 'ts_code', 'alpha99']]



def alpha100(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha100: STD(VOLUME,20)
    计算成交量在20期内的标准差
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['alpha100'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=20, min_periods=1).std())
    return df[['trade_date', 'ts_code', 'alpha100']]


def alpha101(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha101: 
    ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))), RANK(VOLUME), 11))) * -1)
    比较两组相关性排名并根据比较结果计算因子
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 计算 MEAN(VOLUME, 30)
    df['mean_volume_30'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    # 计算 SUM(mean_volume_30, 37)
    df['sum_mean_volume'] = df.groupby('ts_code')['mean_volume_30'].transform(lambda x: x.rolling(window=37, min_periods=1).sum())
    # 计算 CORR(CLOSE, sum_mean_volume, 15)
    df['corr_1'] = (
    df.sort_values(['ts_code', 'trade_date'])
      .groupby('ts_code', group_keys=False)
      .apply(lambda g: g['close'].rolling(window=15, min_periods=1)
                        .corr(g['sum_mean_volume']))
      .reset_index(level=0, drop=True)
)   # 计算 RANK(corr_1)
    df['rank_corr_1'] = df.groupby('trade_date')['corr_1'].rank()
    # 计算 ((HIGH * 0.1) + (VWAP * 0.9))
    df['weighted_price'] = df['high'] * 0.1 + df['vwap'] * 0.9
    # 计算 RANK(weighted_price)
    df['rank_weighted_price'] = df.groupby('trade_date')['weighted_price'].rank()
    # 计算 RANK(VOLUME)
    df['rank_volume'] = df.groupby('trade_date')['volume'].rank()
    # 计算 CORR(rank_weighted_price, rank_volume, 11)
    df['corr_2'] = (
    df.sort_values(['ts_code', 'trade_date'])
      .groupby('ts_code', group_keys=False)
      .apply(lambda g: g['rank_weighted_price'].rolling(window=11, min_periods=1)
                        .corr(g['rank_volume']))
      .reset_index(level=0, drop=True)
    )
    # 计算 RANK(corr_2)
    df['rank_corr_2'] = df.groupby('trade_date')['corr_2'].rank()
    # 比较 rank_corr_1 和 rank_corr_2 并计算因子
    df['alpha101'] = (df['rank_corr_1'] < df['rank_corr_2']) * -1
    return df[['trade_date', 'ts_code', 'alpha101']]


def alpha102(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha102: SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    计算成交量变化的平滑比率指标
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 计算 VOLUME - DELAY(VOLUME, 1)
    df['prev_volume'] = df.groupby('ts_code')['volume'].shift(1)
    df['volume_diff'] = df['volume'] - df['prev_volume']
    # 计算 MAX(volume_diff, 0)
    df['max_volume_diff'] = df['volume_diff'].clip(lower=0)
    # 计算 SMA(max_volume_diff, 6, 1)
    df['sma_1'] = df.groupby('ts_code')['max_volume_diff'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    # 计算 ABS(volume_diff)
    df['abs_volume_diff'] = abs(df['volume_diff'])
    # 计算 SMA(abs_volume_diff, 6, 1)
    df['sma_2'] = df.groupby('ts_code')['abs_volume_diff'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    # 计算最终因子
    df['alpha102'] = (df['sma_1'] / df['sma_2']) * 100
    return df[['trade_date', 'ts_code', 'alpha102']]


def alpha103(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha103: ((20 - LOWDAY(LOW,20))/20)*100
    计算最低价在20期内的低点距离情况并转化为百分比
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 计算 LOWDAY(LOW, 20)，即当前最低价在过去20天中的第几天出现（从0开始计数，0表示当天是20天内最低价）
    def low_day(series):
        return series.rolling(window=20, min_periods=1).apply(
            lambda x: (x == x.min()).argmin(), raw=True
        )
    df['low_day'] = df.groupby('ts_code')['low'].transform(low_day)
    # 计算 (20 - low_day)/20 * 100
    df['alpha103'] = ((20 - df['low_day']) / 20) * 100
    return df[['trade_date', 'ts_code', 'alpha103']]


def alpha104(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha104: (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))
    计算相关性变化量与收盘价标准差排名的乘积并取负
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 计算 CORR(HIGH, VOLUME, 5)
    df['corr_high_volume'] = (
    df.sort_values(['ts_code', 'trade_date'])
      .groupby('ts_code', group_keys=False)
      .apply(lambda g: g['high'].rolling(window=5, min_periods=1).corr(g['volume']))
      .reset_index(level=0, drop=True)
    )
    # 计算 DELTA(corr_high_volume, 5)
    df['delta_corr'] = df.groupby('ts_code')['corr_high_volume'].transform(lambda x: x.diff(5))
    # 计算 STD(CLOSE, 20)
    df['std_close_20'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(window=20, min_periods=1).std())
    # 计算 RANK(std_close_20)
    df['rank_std_close'] = df.groupby('trade_date')['std_close_20'].rank()
    df['alpha104'] = -1 * (df['delta_corr'] * df['rank_std_close'])
    return df[['trade_date', 'ts_code', 'alpha104']]


def alpha105(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha105: (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
    计算开盘价排名与成交量排名在10期内的相关性并取负
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 计算 RANK(OPEN)
    df['rank_open'] = df.groupby('trade_date')['open'].rank()
    # 计算 RANK(VOLUME)
    df['rank_volume'] = df.groupby('trade_date')['volume'].rank()
    # 计算 CORR(rank_open, rank_volume, 10)
    df['corr_rank'] = (
        df.sort_values(['ts_code', 'trade_date'])
        .groupby('ts_code', group_keys=False)
        .apply(lambda g: g['rank_open'].rolling(window=10, min_periods=1)
                            .corr(g['rank_volume']))
      .reset_index(level=0, drop=True)
)

    df['alpha105'] = -1 * df['corr_rank']
    return df[['trade_date', 'ts_code', 'alpha105']]


def alpha106(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha106: CLOSE - DELAY(CLOSE, 20)
    计算收盘价与20期前收盘价的差值
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['delay_close_20'] = df.groupby('ts_code')['close'].shift(20)
    df['alpha106'] = df['close'] - df['delay_close_20']
    return df[['trade_date', 'ts_code', 'alpha106']]


def alpha107(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha107: (-1 * RANK(OPEN - DELAY(HIGH,1)) * RANK(OPEN - DELAY(CLOSE,1)) * RANK(OPEN - DELAY(LOW,1)))
    开盘价与前高/前收/前低差值的排名乘积取负
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 计算延迟指标
    df['delay_high1'] = df.groupby('ts_code')['high'].shift(1)
    df['delay_close1'] = df.groupby('ts_code')['close'].shift(1)
    df['delay_low1'] = df.groupby('ts_code')['low'].shift(1)
    # 计算差值
    df['diff_high'] = df['open'] - df['delay_high1']
    df['diff_close'] = df['open'] - df['delay_close1']
    df['diff_low'] = df['open'] - df['delay_low1']
    # 计算排名
    df['rank_high'] = df.groupby('trade_date')['diff_high'].rank()
    df['rank_close'] = df.groupby('trade_date')['diff_close'].rank()
    df['rank_low'] = df.groupby('trade_date')['diff_low'].rank()
    # 组合计算
    df['alpha107'] = -1 * df['rank_high'] * df['rank_close'] * df['rank_low']
    return df[['trade_date', 'ts_code', 'alpha107']]


def alpha108(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha108: (RANK(HIGH - MIN(HIGH,2)) * RANK(CORR(VWAP, MEAN(VOLUME,120),6))) * -1
    高价短期波动排名与量价相关性排名的乘积取负
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 计算2期内最高价最小值
    df['min_high2'] = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(window=2, min_periods=1).min())
    df['high_diff'] = df['high'] - df['min_high2']
    df['rank_high'] = df.groupby('trade_date')['high_diff'].rank()
    # 计算成交量120期均值与VWAP的相关性
    df['mean_volume120'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=120, min_periods=1).mean())
    df['corr_vwap_vol'] = (
    df.groupby('ts_code', group_keys=False)
      .apply(lambda g: g['vwap'].rolling(window=6, min_periods=1)
                        .corr(g['mean_volume120']))
      .reset_index(level=0, drop=True)
    )
    df['rank_corr'] = df.groupby('trade_date')['corr_vwap_vol'].rank()
    # 组合计算
    df['alpha108'] = -1 * df['rank_high'] * df['rank_corr']
    return df[['trade_date', 'ts_code', 'alpha108']]


def alpha109(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha109: SMA(HIGH-LOW,10,2) / SMA(SMA(HIGH-LOW,10,2),10,2)
    价格波动的双重平滑比率
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()

    # 1) ensure numeric to avoid silent NaNs from string dtypes
    for c in ['high', 'low']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 2) high-low spread
    df['hl_diff'] = df['high'] - df['low']

    # 3) SMA(n=10, m=2) == EWMA with alpha = m/n = 0.2
    alpha = 2 / 10

    g = df.groupby('ts_code', group_keys=False)
    df['sma1'] = g['hl_diff'].apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    df['sma2'] = g['sma1']  .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean())

    # 4) ratio; avoid 0/0 -> NaN
    denom = df['sma2'].replace(0, np.nan)
    df['alpha109'] = df['sma1'] / denom

    return df[['trade_date', 'ts_code', 'alpha109']]


def alpha110(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha110: SUM(MAX(0, HIGH-DELAY(CLOSE,1)),20) / SUM(MAX(0, DELAY(CLOSE,1)-LOW),20) * 100
    20期内上涨动能与下跌动能比率
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 前一日收盘价
    df['delay_close1'] = df.groupby('ts_code')['close'].shift(1)
    # 上涨动能（高价高于前收部分）
    df['up_energy'] = df.apply(lambda row: max(0, row['high'] - row['delay_close1']), axis=1)
    # 下跌动能（前收高于低价部分）
    df['down_energy'] = df.apply(lambda row: max(0, row['delay_close1'] - row['low']), axis=1)
    # 20期累计
    df['sum_up'] = df.groupby('ts_code')['up_energy'].transform(lambda x: x.rolling(window=20, min_periods=1).sum())
    df['sum_down'] = df.groupby('ts_code')['down_energy'].transform(lambda x: x.rolling(window=20, min_periods=1).sum())
    # 比率计算
    df['alpha110'] = (df['sum_up'] / df['sum_down']) * 100
    return df[['trade_date', 'ts_code', 'alpha110']]


def alpha111(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha111: SMA(VOL*( (CLOSE-LOW)-(HIGH-CLOSE) )/(HIGH-LOW),11,2) - SMA(VOL*( (CLOSE-LOW)-(HIGH-CLOSE) )/(HIGH-LOW),4,2)
    量价动能的短期与长期平滑差值
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 计算量价因子核心
    df['hl_diff'] = df['high'] - df['low']
    df['price_bias'] = (df['close'] - df['low']) - (df['high'] - df['close'])
    df['vol_bias'] = df['volume'] * df['price_bias'] / df['hl_diff'].replace(0, 1e-6)  # 避免除零
    # 不同周期SMA
    df['sma11'] = df.groupby('ts_code')['vol_bias'].transform(lambda x: x.rolling(window=11, min_periods=1).mean())
    df['sma4'] = df.groupby('ts_code')['vol_bias'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    # 差值计算
    df['alpha111'] = df['sma11'] - df['sma4']
    return df[['trade_date', 'ts_code', 'alpha111']]


def alpha112(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha112: (SUM(UP,12) - SUM(DOWN,12)) / (SUM(UP,12) + SUM(DOWN,12)) 
    其中UP=当CLOSE>DELAY(CLOSE,1)时为CLOSE-DELAY(CLOSE,1)否则0
    DOWN=当CLOSE<DELAY(CLOSE,1)时为ABS(CLOSE-DELAY(CLOSE,1))否则0
    12期涨跌动能平衡指标
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 前一日收盘价
    df['delay_close1'] = df.groupby('ts_code')['close'].shift(1)
    # 上涨/下跌幅度
    df['up'] = df.apply(lambda row: row['close'] - row['delay_close1'] if row['close'] > row['delay_close1'] else 0, axis=1)
    df['down'] = df.apply(lambda row: abs(row['close'] - row['delay_close1']) if row['close'] < row['delay_close1'] else 0, axis=1)
    # 12期累计
    df['sum_up'] = df.groupby('ts_code')['up'].transform(lambda x: x.rolling(window=12, min_periods=1).sum())
    df['sum_down'] = df.groupby('ts_code')['down'].transform(lambda x: x.rolling(window=12, min_periods=1).sum())
    # 比率计算
    df['alpha112'] = (df['sum_up'] - df['sum_down']) / (df['sum_up'] + df['sum_down']).replace(0, 1e-6)
    return df[['trade_date', 'ts_code', 'alpha112']]


def alpha113(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha113: (-1 * (RANK(SUM(DELAY(CLOSE,5),20)/20) * CORR(CLOSE, VOLUME,2) * RANK(CORR(SUM(CLOSE,5), SUM(CLOSE,20),2)))
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()

    # 5期延迟收盘价的20期累计
    df['delay_close5'] = df.groupby('ts_code')['close'].shift(5)
    df['sum_delay5'] = df.groupby('ts_code')['delay_close5'].transform(
        lambda x: x.rolling(window=20, min_periods=1).sum()
    )
    # 如果你想规范化 rank，最好除以当日有效样本数，而不是固定 20
    # 先按天计算样本数
    counts_by_day = df.groupby('trade_date')['sum_delay5'].transform('count')
    df['rank_sum'] = df.groupby('trade_date')['sum_delay5'].rank() / counts_by_day

    # 收盘价与成交量 2 期相关性 —— 用 apply 而不是 transform
    df['corr_cv'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['close'].rolling(window=2, min_periods=1)
                        .corr(g['volume']))
          .reset_index(level=0, drop=True)
    )

    # 5期 & 20期收盘价和
    df['sum_close5'] = df.groupby('ts_code')['close'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    df['sum_close20'] = df.groupby('ts_code')['close'].transform(
        lambda x: x.rolling(window=20, min_periods=1).sum()
    )

    # 两个和的 2 期相关性 —— 同样用 apply
    df['corr_sum'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['sum_close5'].rolling(window=2, min_periods=1)
                        .corr(g['sum_close20']))
          .reset_index(level=0, drop=True)
    )

    # 截面排名
    df['rank_corr_sum'] = df.groupby('trade_date')['corr_sum'].rank()

    # 组合
    df['alpha113'] = -1 * df['rank_sum'] * df['corr_cv'] * df['rank_corr_sum']

    return df[['trade_date', 'ts_code', 'alpha113']]


def alpha114(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha114: RANK(DELAY(HIGH-OPEN,5)) * RANK( (SUM(CLOSE,5)/5) / (VWAP - CLOSE) )
    高价开盘差延迟排名与价格均值偏离排名的乘积
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 5期延迟的高低开盘差
    df['delay_ho5'] = df.groupby('ts_code')['high'].shift(5) - df.groupby('ts_code')['open'].shift(5)
    df['rank_ho'] = df.groupby('trade_date')['delay_ho5'].rank()
    # 5期均价与当前价差比率
    df['mean_close5'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['price_ratio'] = df['mean_close5'] / (df['vwap'] - df['close']).replace(0, 1e-6)
    df['rank_ratio'] = df.groupby('trade_date')['price_ratio'].rank()
    # 组合计算
    df['alpha114'] = df['rank_ho'] * df['rank_ratio']
    return df[['trade_date', 'ts_code', 'alpha114']]


def alpha115(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha115: RANK(CORR(HIGH*0.9 + CLOSE*0.1, MEAN(VOLUME,30),10)) * RANK(CORR(RANK((HIGH+LOW)/2), TSRANK(VOLUME,10),7))
    加权价格与成交量相关性排名的乘积
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()

    # 加权价格（90%高价 + 10%收盘价）
    df['weighted_price'] = df['high'] * 0.9 + df['close'] * 0.1
    df['mean_volume30'] = df.groupby('ts_code')['volume'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )

    # 第一部分: CORR(weighted_price, mean_volume30, 10)
    df['corr_pv'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['weighted_price'].rolling(window=10, min_periods=1)
                        .corr(g['mean_volume30']))
          .reset_index(level=0, drop=True)
    )
    df['rank_corr1'] = df.groupby('trade_date')['corr_pv'].rank()

    # 第二部分: CORR(RANK((HIGH+LOW)/2), TSRANK(VOLUME,10), 7)
    df['hl_avg'] = (df['high'] + df['low']) / 2
    df['rank_hl'] = df.groupby('trade_date')['hl_avg'].rank()
    df['tsrank_vol'] = df.groupby('trade_date')['volume'].rank(pct=True)

    df['corr_ranks'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['rank_hl'].rolling(window=7, min_periods=1)
                        .corr(g['tsrank_vol']))
          .reset_index(level=0, drop=True)
    )

    df['rank_corr2'] = df.groupby('trade_date')['corr_ranks'].rank()

    # 组合计算
    df['alpha115'] = df['rank_corr1'] * df['rank_corr2']

    return df[['trade_date', 'ts_code', 'alpha115']]


def alpha116(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha116: REGBETA(CLOSE, SEQUENCE,20)
    收盘价对时间序列的20期回归系数（beta）
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    
    def reg_beta(series):
        """计算滚动窗口内收盘价对序列[1,2,...,window]的回归beta"""
        window = 20
        beta_series = []
        for i in range(len(series)):
            if i < window - 1:
                beta_series.append(np.nan)
                continue
            x = np.arange(1, window + 1)  # 时间序列1-20
            y = series.iloc[i - window + 1:i + 1].values
            beta = np.polyfit(x, y, 1)[0]  # 一阶拟合的斜率即beta
            beta_series.append(beta)
        return pd.Series(beta_series, index=series.index)
    
    df['alpha116'] = df.groupby('ts_code')['close'].transform(reg_beta)
    return df[['trade_date', 'ts_code', 'alpha116']]


def alpha117(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha117: TSRANK(VOLUME,32) * (1 - TSRANK((CLOSE+HIGH)-LOW,16)) * (1 - TSRANK(RET,32))
    成交量排名与价格波动排名的复合因子（RET为日收益率）
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 成交量32期百分比排名
    df['tsrank_vol'] = df.groupby('trade_date')['volume'].rank(pct=True)
    # 价格波动指标：(收盘价+高价)-低价
    df['price_vol'] = df['close'] + df['high'] - df['low']
    df['tsrank_pv'] = df.groupby('trade_date')['price_vol'].rank(pct=True)
    # 日收益率（RET）
    df['ret'] = df.groupby('ts_code')['close'].pct_change(1)
    df['tsrank_ret'] = df.groupby('trade_date')['ret'].rank(pct=True)
    # 组合计算
    df['alpha117'] = df['tsrank_vol'] * (1 - df['tsrank_pv']) * (1 - df['tsrank_ret'])
    return df[['trade_date', 'ts_code', 'alpha117']]


def alpha118(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha118: SUM(HIGH-OPEN,20) / SUM(OPEN-LOW,20) * 100
    20期内高开幅度与开低幅度比率
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 计算每日高开和开低幅度
    df['ho_diff'] = df['high'] - df['open']
    df['ol_diff'] = df['open'] - df['low']
    # 20期累计
    df['sum_ho'] = df.groupby('ts_code')['ho_diff'].transform(lambda x: x.rolling(window=20, min_periods=1).sum())
    df['sum_ol'] = df.groupby('ts_code')['ol_diff'].transform(lambda x: x.rolling(window=20, min_periods=1).sum())
    # 比率计算
    df['alpha118'] = (df['sum_ho'] / df['sum_ol'].replace(0, 1e-6)) * 100
    return df[['trade_date', 'ts_code', 'alpha118']]


def alpha119(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha119: SUM(MEAN(VOLUME,5),26) * RANK(DECAYLINEAR(CORR(VWAP, VOLUME,5),7))
    成交量平滑累计与量价相关性衰减排名的乘积
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()

    # 5期成交量均值的26期累计
    df['mean_vol5'] = df.groupby('ts_code')['volume'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df['sum_mean_vol'] = df.groupby('ts_code')['mean_vol5'].transform(
        lambda x: x.rolling(window=26, min_periods=1).sum()
    )

    # VWAP 与 成交量 5期相关性（用 apply 而不是 transform）
    df['corr_vwap_vol'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['vwap'].rolling(window=5, min_periods=1)
                        .corr(g['volume']))
          .reset_index(level=0, drop=True)
    )

    # DECAYLINEAR(即指数加权平均)
    df['decay_corr'] = df.groupby('ts_code')['corr_vwap_vol'].transform(
        lambda x: x.ewm(alpha=2 / (7 + 1), adjust=False).mean()
    )

    # 截面排名
    df['rank_decay'] = df.groupby('trade_date')['decay_corr'].rank()

    # 组合计算
    df['alpha119'] = df['sum_mean_vol'] * df['rank_decay']

    return df[['trade_date', 'ts_code', 'alpha119']]


def alpha120(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha120: RANK(VWAP - CLOSE) / RANK(VWAP + CLOSE)
    加权均价与收盘价偏离排名比率
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 计算VWAP与收盘价的差值和和值
    df['vwap_close_diff'] = df['vwap'] - df['close']
    df['vwap_close_sum'] = df['vwap'] + df['close']
    # 排名计算
    df['rank_diff'] = df.groupby('trade_date')['vwap_close_diff'].rank()
    df['rank_sum'] = df.groupby('trade_date')['vwap_close_sum'].rank()
    # 比率计算
    df['alpha120'] = df['rank_diff'] / df['rank_sum'].replace(0, 1e-6)
    return df[['trade_date', 'ts_code', 'alpha120']]


def alpha121(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha121: (RANK(VWAP - MIN(VWAP,12)) * TSRANK(CORR(TSRANK(VWAP,20), TSRANK(MEAN(VOLUME,60),2),18),3)) * -1
    VWAP波动排名与量价排名相关性的复合因子取负
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()

    # VWAP 与 12期最小值差值（截面排名）
    df['min_vwap12'] = df.groupby('ts_code')['vwap'].transform(lambda x: x.rolling(window=12, min_periods=1).min())
    df['vwap_diff'] = df['vwap'] - df['min_vwap12']
    df['rank_vwap'] = df.groupby('trade_date')['vwap_diff'].rank()

    # VWAP “排名” 与 成交量均值 “排名”
    # (你当前实现用的是截面百分位；保持一致，仅修正相关性计算)
    df['tsrank_vwap20'] = df.groupby('trade_date')['vwap'].rank(pct=True)
    df['mean_vol60'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=60, min_periods=1).mean())
    df['tsrank_vol60'] = df.groupby('trade_date')['mean_vol60'].rank(pct=True)

    # CORR(tsrank_vwap20, tsrank_vol60, 18) —— 用 apply 而不是 transform
    df['corr_ranks'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['tsrank_vwap20'].rolling(window=18, min_periods=1)
                        .corr(g['tsrank_vol60']))
          .reset_index(level=0, drop=True)
    )

    # 截面百分位“排名”（保持你原逻辑）
    df['tsrank_corr'] = df.groupby('trade_date')['corr_ranks'].rank(pct=True)

    # 组合
    df['alpha121'] = -1 * df['rank_vwap'] * df['tsrank_corr']

    return df[['trade_date', 'ts_code', 'alpha121']]



def alpha122(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha122: (SMA3 - DELAY(SMA3,1)) / DELAY(SMA3,1)
    其中SMA3 = SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
    收盘价对数的三重平滑增长率
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    # 收盘价对数
    df['log_close'] = np.log(df['close'])
    # 三重SMA
    df['sma1'] = df.groupby('ts_code')['log_close'].transform(lambda x: x.rolling(window=13, min_periods=1).mean())
    df['sma2'] = df.groupby('ts_code')['sma1'].transform(lambda x: x.rolling(window=13, min_periods=1).mean())
    df['sma3'] = df.groupby('ts_code')['sma2'].transform(lambda x: x.rolling(window=13, min_periods=1).mean())
    # 增长率计算
    df['delay_sma3'] = df.groupby('ts_code')['sma3'].shift(1)
    df['alpha122'] = (df['sma3'] - df['delay_sma3']) / df['delay_sma3'].replace(0, 1e-6)
    return df[['trade_date', 'ts_code', 'alpha122']]


import numpy as np
import pandas as pd

def alpha123(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha123: (RANK(CORR(SUM((HIGH+LOW)/2,20), SUM(MEAN(VOLUME,60),20),9)) < RANK(CORR(LOW, VOLUME,6))) * -1
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()

    df['hl_avg'] = (df['high'] + df['low']) / 2
    df['sum_hl20'] = df.groupby('ts_code')['hl_avg'].transform(lambda x: x.rolling(window=20, min_periods=1).sum())
    df['mean_vol60'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=60, min_periods=1).mean())
    df['sum_vol60'] = df.groupby('ts_code')['mean_vol60'].transform(lambda x: x.rolling(window=20, min_periods=1).sum())

    # FIXED: rolling corr via groupby.apply
    df['corr_hl_vol'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['sum_hl20'].rolling(window=9, min_periods=1).corr(g['sum_vol60']))
          .reset_index(level=0, drop=True)
    )
    df['rank_corr1'] = df.groupby('trade_date')['corr_hl_vol'].rank()

    # FIXED: rolling corr via groupby.apply
    df['corr_low_vol'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['low'].rolling(window=6, min_periods=1).corr(g['volume']))
          .reset_index(level=0, drop=True)
    )
    df['rank_corr2'] = df.groupby('trade_date')['corr_low_vol'].rank()

    df['alpha123'] = (df['rank_corr1'] < df['rank_corr2']).astype(int) * -1
    return df[['trade_date', 'ts_code', 'alpha123']]


def alpha124(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha124: (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMA(CLOSE,30)),4)
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['close_vwap_diff'] = df['close'] - df['vwap']
    df['tsma_close30'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df['rank_tsma'] = df.groupby('trade_date')['tsma_close30'].rank()
    df['decay_rank'] = df.groupby('ts_code')['rank_tsma'].transform(lambda x: x.ewm(alpha=2/(4+1), adjust=False).mean())
    df['alpha124'] = df['close_vwap_diff'] / df['decay_rank'].replace(0, 1e-6)
    return df[['trade_date', 'ts_code', 'alpha124']]


def alpha125(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha125: RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,80),17),20)) / RANK(DECAYLINEAR(DELTA((CLOSE*0.5+VWAP*0.5),3),16))
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['mean_vol80'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=80, min_periods=1).mean())

    # FIXED: rolling corr via groupby.apply
    df['corr_vwap_vol'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['vwap'].rolling(window=17, min_periods=1).corr(g['mean_vol80']))
          .reset_index(level=0, drop=True)
    )
    df['decay_corr'] = df.groupby('ts_code')['corr_vwap_vol'].transform(lambda x: x.ewm(alpha=2/(20+1), adjust=False).mean())
    df['rank_decay1'] = df.groupby('trade_date')['decay_corr'].rank()

    df['weighted_price'] = 0.5*df['close'] + 0.5*df['vwap']
    df['delta_price3'] = df.groupby('ts_code')['weighted_price'].transform(lambda x: x.diff(3))
    df['decay_delta'] = df.groupby('ts_code')['delta_price3'].transform(lambda x: x.ewm(alpha=2/(16+1), adjust=False).mean())
    df['rank_decay2'] = df.groupby('trade_date')['decay_delta'].rank()

    df['alpha125'] = df['rank_decay1'] / df['rank_decay2'].replace(0, 1e-6)
    return df[['trade_date', 'ts_code', 'alpha125']]


def alpha126(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha126: (CLOSE + HIGH + LOW) / 3
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['alpha126'] = (df['close'] + df['high'] + df['low']) / 3
    return df[['trade_date', 'ts_code', 'alpha126']]


def alpha127(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha127: SQRT(MEAN(100*(CLOSE - MAX(CLOSE,12))/MAX(CLOSE,12), 2))
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['max_close12'] = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(window=12, min_periods=1).max())
    df['retrace'] = 100 * (df['close'] - df['max_close12']) / df['max_close12'].replace(0, 1e-6)
    df['mean_retrace'] = df.groupby('ts_code')['retrace'].transform(lambda x: x.rolling(window=2, min_periods=1).mean())
    df['alpha127'] = np.sqrt(df['mean_retrace'].abs())
    return df[['trade_date', 'ts_code', 'alpha127']]


def alpha128(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha128: 100 - 100 / (1 + SMA(UP_VOL,14) / SMA(DOWN_VOL,14))
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['hlc_avg'] = (df['high'] + df['low'] + df['close']) / 3
    df['delay_hlc'] = df.groupby('ts_code')['hlc_avg'].shift(1)

    df['up_vol'] = np.where(df['hlc_avg'] > df['delay_hlc'], df['hlc_avg'] * df['volume'], 0.0)
    df['down_vol'] = np.where(df['hlc_avg'] < df['delay_hlc'], df['hlc_avg'] * df['volume'], 0.0)

    df['sma_up'] = df.groupby('ts_code')['up_vol'].transform(lambda x: x.rolling(window=14, min_periods=1).mean())
    df['sma_down'] = df.groupby('ts_code')['down_vol'].transform(lambda x: x.rolling(window=14, min_periods=1).mean())
    df['ratio'] = df['sma_up'] / df['sma_down'].replace(0, 1e-6)
    df['alpha128'] = 100 - 100 / (1 + df['ratio'])
    return df[['trade_date', 'ts_code', 'alpha128']]


def alpha129(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha129: SUM(当CLOSE<DELAY(CLOSE,1)时为ABS(CLOSE-DELAY(CLOSE,1))否则0,12)
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['delay_close1'] = df.groupby('ts_code')['close'].shift(1)
    df['down_amount'] = np.where(df['close'] < df['delay_close1'], (df['close'] - df['delay_close1']).abs(), 0.0)
    df['alpha129'] = df.groupby('ts_code')['down_amount'].transform(lambda x: x.rolling(window=12, min_periods=1).sum())
    return df[['trade_date', 'ts_code', 'alpha129']]


def alpha130(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha130: RANK(DECAYLINEAR(CORR((HIGH+LOW)/2, MEAN(VOLUME,40),9),10)) * RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME),7),3))
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()

    # Part 1
    df['hl_avg'] = (df['high'] + df['low']) / 2
    df['mean_vol40'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=40, min_periods=1).mean())

    # FIXED: rolling corr via groupby.apply
    df['corr_hl_vol'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['hl_avg'].rolling(window=9, min_periods=1).corr(g['mean_vol40']))
          .reset_index(level=0, drop=True)
    )
    df['decay1'] = df.groupby('ts_code')['corr_hl_vol'].transform(lambda x: x.ewm(alpha=2/(10+1), adjust=False).mean())
    df['rank1'] = df.groupby('trade_date')['decay1'].rank()

    # Part 2
    df['rank_vwap'] = df.groupby('trade_date')['vwap'].rank()
    df['rank_vol'] = df.groupby('trade_date')['volume'].rank()

    # FIXED: rolling corr via groupby.apply
    df['corr_ranks'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['rank_vwap'].rolling(window=7, min_periods=1).corr(g['rank_vol']))
          .reset_index(level=0, drop=True)
    )
    df['decay2'] = df.groupby('ts_code')['corr_ranks'].transform(lambda x: x.ewm(alpha=2/(3+1), adjust=False).mean())
    df['rank2'] = df.groupby('trade_date')['decay2'].rank()

    df['alpha130'] = df['rank1'] * df['rank2']
    return df[['trade_date', 'ts_code', 'alpha130']]


def alpha131(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha131: RANK(DELTA(VWAP,1)) * TSRANK(CORR(CLOSE, MEAN(VOLUME,50),18),18)
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()
    df['delta_vwap1'] = df.groupby('ts_code')['vwap'].diff(1)
    df['rank_delta'] = df.groupby('trade_date')['delta_vwap1'].rank()
    df['mean_vol50'] = df.groupby('ts_code')['volume'].transform(lambda x: x.rolling(window=50, min_periods=1).mean())

    # FIXED: rolling corr via groupby.apply
    df['corr_cv'] = (
        df.groupby('ts_code', group_keys=False)
          .apply(lambda g: g['close'].rolling(window=18, min_periods=1).corr(g['mean_vol50']))
          .reset_index(level=0, drop=True)
    )
    df['tsrank_corr'] = df.groupby('trade_date')['corr_cv'].rank(pct=True)

    df['alpha131'] = df['rank_delta'] * df['tsrank_corr']
    return df[['trade_date', 'ts_code', 'alpha131']]


# def alpha132(data: pd.DataFrame) -> pd.DataFrame:
#     """
#     Alpha132: MEAN(AMOUNT,20)
#     """
#     df = data.sort_values(['ts_code', 'trade_date']).copy()
#     df['alpha132'] = df.groupby('ts_code')['amount'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
#     return df[['trade_date', 'ts_code', 'alpha132']]


def alpha133(data: pd.DataFrame) -> pd.DataFrame:
    """
    Alpha133: ((20 - HIGHDAY(HIGH,20))/20)*100 - ((20 - LOWDAY(LOW,20))/20)*100
    """
    df = data.sort_values(['ts_code', 'trade_date']).copy()

    def high_day(series):
        return series.rolling(window=20, min_periods=1).apply(lambda x: (x == x.max()).argmax(), raw=True)

    def low_day(series):
        return series.rolling(window=20, min_periods=1).apply(lambda x: (x == x.min()).argmin(), raw=True)

    df['high_day'] = df.groupby('ts_code')['high'].transform(high_day)
    df['low_day'] = df.groupby('ts_code')['low'].transform(low_day)
    df['high_pct'] = ((20 - df['high_day']) / 20) * 100
    df['low_pct'] = ((20 - df['low_day']) / 20) * 100
    df['alpha133'] = df['high_pct'] - df['low_pct']
    return df[['trade_date', 'ts_code', 'alpha133']]
