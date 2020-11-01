import os
import sys
import subprocess
import urllib.request
import pandas as pd
import time

def extractor(site):
    '''Extract tables from a single or a list of urls or html filenames passed.'''
    header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
      "X-Requested-With": "XMLHttpRequest"}
    
    try:
        time_out = time.process_time() + 120 # 2 minutes timeout

        while time.process_time() <= time_out:
            if len(site.split('//'))>1:
                fname = site.split('//')[1]
            else:
                fname = site
                site = 'http://'+site

    #     print('Extracting tables from: ' + site)


            req=urllib.request.Request(site, headers=header)
            content = urllib.request.urlopen(req).read()
            df1= pd.read_html(content)
            return df1, ''
        else:
            raise Exception('timeout')
    except Exception as e:
        df1=[]
        return df1, e
