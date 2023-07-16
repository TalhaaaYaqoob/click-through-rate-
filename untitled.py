import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pylab as pl
from sklearn.cluster import KMeans
import hdbscan
import flask
import json
app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def Popularity():
    with open('fakeData.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df1 = pd.DataFrame(df['campaign'].values.tolist())
    df1.columns = 'campaign.'+ df1.columns
    df3 = pd.concat([df, df1],axis=1)
    df3['ctr']=(df3['campaign.clicks']/df3['campaign.impressions'])*100
    df3['score']=5*df3['campaign.clicks']+3*df3['campaign.impressions']+df3['ctr']
    df4=df3.sort_values(['score'],ascending=False)
    n=df4.head(5)
    df5=df4.iloc[5:]
    s=df5.sample(5)
    df6=[n,s]
    df7=pd.concat(df6)
    df8=df7[['name','_id','status','campaign._id','campaign.name','score','ctr']]

    out = df8.to_json(orient='records')[1:-1].replace('},{', '} {')

    
    
    
    
#     df=pd.read_csv("clean.csv")
#     df2=df[['product_name','impressions','clicks','conversions']]
#     df2['ctr']=(df2['clicks']/df2['impressions'])*100
#     kmeans=KMeans(n_clusters=10,random_state=0).fit(df2)
#     clusterer=hdbscan.HDBSCAN(min_cluster_size=10)
#     cluster_labels=clusterer.fit_predict(df2)
#     df2['score']=5*df2['clicks']+3*df2['impressions']+df2['ctr']
#     df3=df2.sort_values(['score'],ascending=False)
#     s=df3.head(5)
#     df4=df3.iloc[5:]
#     n=df4.sample(5)
#     df5=[s,n]
#     df6=pd.concat(df5)
#     out = df6.to_json(orient='records')[1:-1].replace('},{', '} {')
    
    #df7=df6['product_name'].values.tolist()
    #values = ','.join(str(v) for v in df7)

    
    return out  
    

app.run(host='127.0.0.1', port=5003)





    

