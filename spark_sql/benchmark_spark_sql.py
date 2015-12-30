###################################
#
# Benchmark Spark SQL
# J. Gartner
# Read in tsv million song database,
# benchmark querry times
#
###################################

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from datetime import datetime
import argparse
import numpy

def recordToRows(line):
    ll = line.split("\t")
    return Row(
        artist_id = ll[0],
        sample_rate = ll[1],
        artist_familiarity = ll[2],
        artist_hotness = ll[3],
        artist_name = ll[4],
        audio_md5 = ll[5],
        danceability = ll[6],
        duration = ll[7],
        end_of_fade_in = ll[8],
        energy = ll[9],
        key = ll[10],
        key_confidence = ll[11],
        loudness = ll[12],
        mode = ll[13],
        mode_confidence = ll[14],
        release = ll[15],
        song_hotttness = ll[16],
        song_id = ll[17],
        start_of_fade_out = ll[18],
        tempo = ll[19],
        time_signature = ll[20],
        time_signature_confidence = ll[21],
        title = ll[22],
        year = ll[23])

def time_querry(querry_string, sqlContext, method=0):
    times = []
    for i in range(100):
        t0 = datetime.now()
        if method==0:
            n = sqlContext.sql(querry_string).count()
        else:
            n = sqlContext.sql(querry_string).collect()[0]._c0
        times.append((datetime.now()-t0).total_seconds())
    ave_t = sum(times)/float(len(times))
    std = numpy.std(numpy.array(times))
    diff = times[0] - ave_t
    return [ave_t, std, diff, n]

def main(n_part, hdfs_path):
    print "********************\n*"
    print "* Start main\n*"
    print "********************"
    conf = SparkConf().setAppName("Benchmark Spark SQL")
    sc = SparkContext(conf = conf)
    sqlContext = SQLContext(sc)
    rowsRDD = sc.textFile(hdfs_path).repartition(n_part).map(lambda x: recordToRows(x)).cache()
    df = sqlContext.createDataFrame(rowsRDD).cache()
    df.count()
    df.registerTempTable("msd_table")
    print "********************\n*"
    print "* Start querres\n*"
    print "********************"
    [ave_t1, std1, dt1, n1] = time_querry("SELECT * FROM msd_table WHERE msd_table.artist_name = 'Taylor Swift'", sqlContext)
    [ave_t2, std2, dt2, n2] = time_querry("SELECT COUNT(*) FROM msd_table WHERE msd_table.artist_name = 'Taylor Swift'", sqlContext, method=1)
    [ave_t3, std3, dt3, n3] = time_querry("SELECT * FROM msd_table WHERE msd_table.artist_hotness > 0.75", sqlContext)
    [ave_t4, std4, dt4, n4] = time_querry("SELECT COUNT(*) FROM msd_table WHERE msd_table.artist_hotness > 0.75", sqlContext, method=1)
    if n1 != n2:
        print "\t!!!!Error, counts disagree for the number of T.S. songs!"
    if n3 != n4:
        print "\t!!!!Error, counts disagree for the number of high paced songs!"
    print "********************\n*"
    print "* Results"
    print "\t".join(map(lambda x: str(x), [ave_t1, std1, dt1, ave_t2, std2, dt2, ave_t3, std3, dt3, ave_t4, std4, dt4]))
    print "********************"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("partitions", help="Number of partitions")
    parser.add_argument("hdfs_file_path", help="HDFS path to input file")
    args = parser.parse_args()
    n_part = int(args.partitions)
    main(n_part, args.hdfs_file_path)