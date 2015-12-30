import datetime
import elasticsearch

def datetime_to_es_format(date):
    return str(date.date())+"T"+str(date.hour)+":"+str(date.minute)+":"+str(date.second)+"Z"

def main():
    es = elasticsearch.Elasticsearch("http://scc:9200")
    now = datetime.datetime.now()
    start = datetime.datetime(2013, 1, 1)
    end = datetime.datetime(2013, 1, 1, 23, 59)
    for i in range(365):
        this_start = start + datetime.timedelta(i)
        this_end = end + datetime.timedelta(i)
        res = es.search(\
            index="benchmarking", \
            doc_type="post", \
            body={
                "filter": {
                    "bool":{
                        "must" :[
                            {
                                "range": {
                                    "post_date":{
                                        "gte" : datetime_to_es_format(this_start),
                                        "lte" :datetime_to_es_format(this_end)
                                     }
                                }
                            }
                        ]
                    }
                }
            }\
        )
    diff = datetime.datetime.now() - now
    print "difference:", diff

if __name__ == '__main__':
    main()
