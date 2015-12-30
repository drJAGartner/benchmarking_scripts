import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.elasticsearch.spark._
import org.apache.spark.sql._
import java.util.Calendar
import java.text.SimpleDateFormat
import java.io._


val ES_INDEX="benchmarking/post"

// insert list of paths to data sets in hdfs
//val DATASETS = Seq("/data/geqe/HighPlains.parquet")
//val DATASETS = Seq("/data/geqe/Cleveland.parquet")
//val DATASETS = Seq("/data/geqe/HighPlains.parquet", "/data/geqe/Cleveland.parquet", "/data/geqe/summerCampToParquet")
val DATASETS = Seq("/data/geqe/northeast95.parquet")
//val DATASETS = Seq("/data/geqe/HighPlains.parquet", "/data/geqe/Cleveland.parquet", "/data/geqe/summerCampToParquet", "/data/geqe/northeast95.parquet", "/data/geqe/Texas.parquet")


// get todays date as yyyy-MM-dd
val indexedDate = new SimpleDateFormat("yyyy-MM-dd").format(Calendar.getInstance().getTime())


// these lines are commented out to run this script from the spark-shell interactively
// if you want to compile this program to a jar and run it that way you'll need to uncomment these lines

//val conf = new SparkConf().setAppName("geqe es-ingest")
//val sc = new SparkContext(conf)
//val sqlContext = new org.apache.spark.sql.SQLContext(sc)

// bring in to the saveToEs function from the elastic search hadoop jar
import sqlContext.implicits._

val now = Calendar.getInstance().getTime()
for (dataset <- DATASETS) {
  println("Processing dataset: "+dataset)
  sqlContext.parquetFile(dataset)
    .repartition(25)
    .map({ case Row(dt:Any,img:String,lat:Double,lon:Double,source:String,text:String,user:String) =>
      var postDate = dt.toString().replace(" ","T").split("\\.")(0)
      Map("source" -> source, "imageUrl" -> img, "indexedDate" -> indexedDate, "dataset"-> dataset, "user" -> user, "post_date" -> postDate, "message" -> text, "location" -> Map("type" ->"point", "coordinates"-> List(lon,lat)))
    }).saveToEs(ES_INDEX)
}
val now2 = Calendar.getInstance().getTime()
val diff = now2.getTime() - now.getTime()



