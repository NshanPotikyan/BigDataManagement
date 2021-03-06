package bdm_com.spark.scala.bdm_demo

import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector

import scala.math.{pow, sqrt}



object Reducer{

  class Reduce(trainRDD: scala.collection.Map[Long,LabeledPoint],
               testRDD: scala.collection.Map[Long,LabeledPoint],
               dist: RDD[(Long, List[List[Double]])] ,
               K: Int
              ) extends Serializable {
    
    // keeping first K values
    val topK = dist.mapValues(Values =>
      Values.take(K))

    val groupedTopK = topK.mapValues(Values => CentroidByClass(Values))


    val result = groupedTopK.map(keyValue => (keyValue._1,
      keyValue._2.mapValues(
        mu => DistCentroid(mu,keyValue._1.toInt)
      )
    )).mapValues(pair => chooseMinDist(pair))


    def chooseMinDist(meanDist: Map[Double,Double]): Double ={
      val comb = meanDist.minBy{ case (key, value) => value }
      comb._1
     }

    def CentroidByClass(lists: List[List[Double]]): Map[Double, Array[Double]] = {

      // Grouping the train ids per label:
      val groupByClass = lists.map(
        list => (list.apply(2), list.head))
        // groupByKey alternative
        .groupBy(_._1)
        .map { case (k, v) => k -> v.map {
          _._2.toInt
        }
        }

      groupByClass.mapValues(v => centroid(v))

      }

    def DistCentroid(centroid: Array[Double],
                     ts_id: Int): Double = {

      val test_feature = testRDD.get(ts_id).get.features.toArray

      distance(centroid,test_feature)


    }


    def distance(xs: Array[Double],
                 ys: Array[Double]) = {
      sqrt((xs zip ys).map {
        case (x, y) => pow(y - x, 2) }.sum)
    }


    def centroid(vec: List[Int]): Array[Double] = {

      val centre = trainRDD.filter(
        row => vec.contains(row._1)
      ).map(par => (par._2.features, 1))
        .reduce((a, b) => FeatureSum(a, b))

      val mu = Vectors.dense(centre._1.toArray).toArray.map(_ / centre._2)

      mu

    }


    def FeatureSum(tuple: (Vector, Int),
                   tuple1: (Vector, Int)): (Vector, Int) = {

      val sum = Vectors.dense((tuple._1.toArray zip tuple1._1.toArray).map {
        case (x, y) => x + y
      }
      )

      val Z = tuple._2 + tuple1._2

      (sum, Z)

    }


    }



}