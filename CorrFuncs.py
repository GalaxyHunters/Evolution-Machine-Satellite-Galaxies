from PIL import Image
import random
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats as st
from MiscFuncs import df_stats, get_concat_h
path_1 = r'/vol/sci/astro/home/adambeilialpha/'

def HistGraphLoop(sgal_features):
    for i,x in sgal_features.iteritems():
        print i
        b=np.log10(sgal_features[i].fillna(0.0).astype(float))
        plt.figure()
        plat_normed = sns.distplot(b.replace([np.inf, -np.inf], np.nan).dropna()).set_title('Normalized Hist')
        plat_normed.figure.savefig(path_1+r'/sgal_features_temp_1.png')
        plt.figure()
        fat = sns.distplot(sgal_features[i].dropna(),color="y").set_title('OG Hist')
        fat.figure.savefig(path_1+r'/sgal_features_temp_2.png')
        im1 = Image.open(path_1+r'/sgal_features_temp_1.png')
        im2 = Image.open(path_1+r'/sgal_features_temp_2.png')

        get_concat_h(im1, im2).save(path_1+r'/Hist_Normed_vs_OG_feature_number__'+str(a)+'.png')
        
def HistCorrelation(sgal_features,a,b,feature_name):
    zero_mask = sgal_features[a]==0; sgal_features[a][zero_mask] = 1.
    zero_mask = sgal_features[b]==0; sgal_features[b][zero_mask] = 1.
    x1=np.array(np.log10(sgal_features[a].fillna(1.0).astype(float)))
    x2=np.array(np.log10(sgal_features[b].fillna(1.0).astype(float)))
    data = {a: x1,b: x2}
    df=pd.DataFrame(columns=[a,b],data=data)
    plt.figure()
    hello1 = sns.lmplot(x=a,y=b,data=df).set(title='Normalized Linear Correlation')
    
    y1=np.array(sgal_features[a].fillna(0.0).astype(float))
    y2=np.array(sgal_features[b].fillna(0.0).astype(float))
    data_2 = {a: y1,b: y2}
    df=pd.DataFrame(columns=[a,b],data=data_2)
    plt.figure()
    hello2 = sns.lmplot(x=a,y=b,data=df,palette ='ch:2.5,-.2,dark=.3').set(title='OG Linear Correlation')
    
    hello1.savefig(path_1+r'/HistCorrelationTemp.png')
    hello2.savefig(path_1+r'/HistCorrelationTemp2.png')
    
    im1 = Image.open(path_1+r'/HistCorrelationTemp.png')
    im2 = Image.open(path_1+r'/HistCorrelationTemp2.png')

    get_concat_h(im1, im2).save(path_1+r'/Hist_Normed_Correlation_example_'+feature_name+'.png')
    
def CorrDensityPlots(df,y):
    df = df.fillna(0.0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.reset_index().drop(['tgid'],axis=1)
    plt.figure()
    feature_list = df.drop(['qlabel'],axis=1).keys().values
    fig, axs = plt.subplots(len(feature_list), 1,figsize=(20,y),sharey=True)
    #sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":10}) 
    sns.set(font_scale=1.)

    for feature in feature_list:
        n=feature_list.tolist().index(feature)
        try:
            sns.kdeplot(df[feature],df['qlabel'],shade=True,shade_lowest=False,ax=axs[n])
        except:
            'Some Error'
        continue
def CorrDensityPlots_normed(df,y):
    df = df.fillna(0.0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.reset_index().drop(['tgid'],axis=1)
    
    plt.figure()
    feature_list = df.drop(['qlabel'],axis=1).keys().values
    fig, axs = plt.subplots(len(feature_list), 1,figsize=(12,y),sharey=True)
    #sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":10}) 

    for feature in feature_list:
        n=feature_list.tolist().index(feature)
        zero_mask = df[feature]==0; df[feature][zero_mask]=1.
        feature_normed = np.log10(df[feature].fillna(1.0).astype(float))
        new_df = pd.DataFrame(index=df.index,data={feature:feature_normed,'qlabel':df['qlabel']})
        new_df = new_df.fillna(0.0)
        new_df = new_df.replace([np.inf, -np.inf], np.nan).dropna()
        try:
            sns.kdeplot(new_df[feature],new_df['qlabel'],shade=True,shade_lowest=False,ax=axs[n])#,data=df,palette ='ch:2.5,-.2,dark=.3').set(title='OG Linear Correlation')
        except:
            print 'np.LinAlg.Error'
        continue    

def df_normed(df):
    feature_list = df.drop(['qlabel'],axis=1).keys().values
    new_df = pd.DataFrame(index=df.index,data=df['qlabel'])
    for feature in feature_list:
        try:
            n=feature_list.tolist().index(feature)
            zero_mask = df[feature]==0; df[feature][zero_mask]=1.
            feature_normed = np.log10(df[feature].fillna(1.0).astype(float))
            new_df[feature] = feature_normed
        except:
            print('Feature could not be normalized', feature)
    return new_df
        
        
def PearsonCorr(df):
    sns.set(style="white")

    df = df.fillna(0.0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.reset_index().drop(['tgid'],axis=1)
    print len(df)
    # Compute the correlation matrix
    corr = df.corr(method='pearson')

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    plot=sns.heatmap(corr, mask=mask, cmap='PuOr', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .9})
    sns.set(font_scale=1.5)
    plot.figure.savefig(path_1+r'/plc_1.png')
    mask=(corr['qlabel']**2)>=0.1
    important_features = corr['qlabel'][mask]
    print(important_features)
    