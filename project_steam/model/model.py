import sqlite3
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))



def model(df, user_id):
    

    df['Id'] = df.Id.astype(str)
    steam_df = df.copy()
    steam_df['like'] = [1 if x > 40 else 0 for x in steam_df['PlayTime']]
    # print(df)

    # x = df.groupby(['Id', 'Title'])['Behavior_Name'].size()
    # s = x[x == 1]
    # print(len(s))
    # print(len(x))

    # 구매는 했지만 플레이 안한 사람 필터링
    boolean_index = steam_df.groupby(['Id','Title'])['Behavior_Name'].transform('size') < 2

    steam_df.loc[boolean_index,'PlayTime'] = 0
    # print(df.loc[df['PlayTime']==0])


    steam_df.loc[steam_df.PlayTime==0,'Behavior_Name'] = 'play'

    # print(df.loc[df['PlayTime'] ==0])
    steam_df = steam_df[steam_df.Behavior_Name != 'purchase']

    #인기도와 다르게 만족도로 바꾼 매트릭스
    d = {'like':'Sum Likes','PlayTime':'Avg Hours Played'}
    metrics_df = steam_df.groupby(['Title'], as_index=False).agg({'like':'sum','PlayTime':'mean'}).rename(columns=d)

    # print(metrics_df.loc[metrics_df['Title'] == "Dota 2"]) #도타2로 체크

    # 평균 플레이시간
    c = metrics_df['Avg Hours Played'].mean()

    # 최소 좋아요 수
    m = metrics_df['Sum Likes'].quantile(0.95)

    # 게임 별 평가점수 만들기
    def make_score(steam_df, m=m, C=c):
        l = steam_df['Sum Likes']
        a = steam_df['Avg Hours Played']
        return (l/(l+m) * a) + (m/(l+m) * C)

    metrics_df['score'] = metrics_df.apply(make_score, axis=1)
    # print(metrics_df.sort_values(by='score',ascending=False))



    games_df = pd.DataFrame(steam_df.Title.unique(), columns=['Title'])
    games_df['index_col'] = games_df.index
    games_df
    steam_df = steam_df.merge(games_df, on='Title')

    usergroup = steam_df.groupby('Id')
    usergroup.head()

    noOfUsers = 1000

    train_list = []

    i = 0
    for userID, cur in usergroup:
        temp = [0]*len(games_df)
        # For each game in list
        for no, game in cur.iterrows():
            temp[game['index_col']] = game['PlayTime']
            i+=1
        train_list.append(temp)
        
        if noOfUsers == 0:
            break
        noOfUsers -= 1


    hiddenUnits = 50
    visibleUnits = len(df['Title'].unique())
    vb = tf.placeholder(tf.float32, [visibleUnits])  
    hb = tf.placeholder(tf.float32, [hiddenUnits]) 
    W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits]) 

    # Phase 1: Input Processing
    v0 = tf.placeholder("float", [None, visibleUnits])
    _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  
    h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0)))) 

    # Phase 2: Reconstruction
    _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb) 
    v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
    h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

    # Learning rate
    alpha = 1
    
    # Create the gradients
    w_pos_grad = tf.matmul(tf.transpose(v0), h0)
    w_neg_grad = tf.matmul(tf.transpose(v1), h1)

    # Calculate the Contrastive Divergence to maximize
    CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

    # Create methods to update the weights and biases
    update_w = W + alpha * CD
    update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
    update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

    # Set the error function, here we use Mean Absolute Error Function
    err = v0 - v1
    err_sum = tf.reduce_mean(err*err)

    # print(err_sum)

    cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

    cur_vb = np.zeros([visibleUnits], np.float32)

    cur_hb = np.zeros([hiddenUnits], np.float32)

    prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

    prv_vb = np.zeros([visibleUnits], np.float32)

    prv_hb = np.zeros([hiddenUnits], np.float32)
    sess = tf.Session()

    inputUser = [train_list[150]]
    hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
    feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
    rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})

    # 가장 추천하는 10개 리스트
    inputuser_games = games_df
    inputuser_games["Recommendation Score"] = rec[0]
    # print(inputuser_games.sort_values(["Recommendation Score"], ascending=False).head(10))

    #유저 정보
    userid = user_id
    #유저정보 찾기
    muser_df = steam_df.loc[(steam_df['Id'] == userid) & (steam_df['PlayTime'] >0)]
    # print(muser_df)

    df_all = inputuser_games.merge(muser_df, how='left', indicator=True)
    unplayed_games = df_all[df_all['_merge']=='left_only']

    #추천 상위 5개
    return (unplayed_games.loc[:,['Title','Recommendation Score']].sort_values(['Recommendation Score'], ascending=False).head(5))