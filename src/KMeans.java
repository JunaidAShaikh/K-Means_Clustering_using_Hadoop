import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.util.*;
import java.io.*;
import org.apache.hadoop.fs.FileSystem;


public class KMeans {

	static int numAttr=0;
	static int iteration=0;
	static Map<Integer,Double[]>centroid = new HashMap<Integer,Double[]>(); 
	static Map<Integer,Double[]>previous = null; 
	static Map<Integer,Integer> RowToCentroid = new TreeMap<Integer,Integer>(); 
	static Map<Integer, Integer> RowToGround = new TreeMap<Integer, Integer>();

	public static class MyMapper extends Mapper<Object, Text, IntWritable, Text>{

		private Text pointText = new Text();

		public static Double euclidDist(Double[] point1, Double[] point2) {
			Double distance = 0.0;
			Double axisDist =0.0;
			for (int i = 0; i < point1.length; i++) {
				axisDist = Math.abs(point1[i] - point2[i]);
				distance += axisDist * axisDist;
			}
			return distance;
		}


		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

			String line = value.toString();


			String[] lineArray = line.split("\\t");
			int row_idx = Integer.parseInt(lineArray[0])-1;
			StringBuilder pointString = new StringBuilder();
			if(iteration==0)
			{
				RowToGround.put(row_idx, Integer.parseInt(lineArray[1])-1);
			}
			numAttr = lineArray.length - 2;
			Double minDist = Double.MAX_VALUE;
			int closest = 0;
			Double[] dataPoint = new Double[numAttr];
			Double distFromCentroid = 0.0;

			for (int i = 0; i < dataPoint.length; i++) 
			{
				dataPoint[i] = Double.parseDouble(lineArray[i + 2]);
				pointString.append(lineArray[i + 2] + "\\t");
			}

			for (int c: centroid.keySet()) 
			{
				distFromCentroid = euclidDist(centroid.get(c), dataPoint);  //get distance of each row from all centroids
				if (distFromCentroid < minDist) 
				{
					minDist = distFromCentroid;
					closest = c;
				}
			}

			IntWritable closestCentroid = new IntWritable(closest);
			pointText.set(line);
			context.write(closestCentroid, pointText);


		}
	}

	public static class MyReducer
	extends Reducer<IntWritable, Text,IntWritable,Text> {

		private Text newCentroidMean = new Text();

		public void reduce(IntWritable clusterId, Iterable<Text> rows, Context context)
				throws IOException, InterruptedException 
		{
			Double count = 0.0;
			Double[] mean = new Double[numAttr];

			Iterator<Text> it =rows.iterator();
			while (it.hasNext()) 
			{
				Text line= it.next();

				String pointString = line.toString();
				String[] pointStringArray = pointString.split("\\t");
				RowToCentroid.put(Integer.parseInt(pointStringArray[0])-1,clusterId.get());

				for (int i = 0; i < numAttr; i++) 
				{
					mean[i] = (mean[i]==null?0.0:mean[i])+Double.parseDouble(pointStringArray[i+2]);
				}
				count++;
			}

			StringBuilder meanString = new StringBuilder();

			for (int i = 0; i < numAttr; i++) 
			{
				mean[i] = mean[i] / count;
				meanString.append(mean[i] + "\\t");
			}

			centroid.put(clusterId.get(),mean);

			//Text temp =new Text(centroid.toString());
			//newCentroidMean.set(meanString.toString());
			//context.write(clusterId, newCentroidMean);
		}
	}

	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();
		String OUT ="/output/";
		String input = "/input_kmeans/";
		String InitialCentroidFolder = "/input_kmeans/";
		String DATA_FILE_NAME = "new_dataset_1.txt";
		String CENTROID_FILE_NAME ="new_dataset_1-center.txt";
		String output = OUT + System.nanoTime();


		boolean randomCentroid = true;
		int preDefCentroids=3;
		int cluster_number=0;
		PriorityQueue<Integer> CentroidIndexes = new PriorityQueue<Integer>();
		String line ="";

		if(randomCentroid)
		{

			cluster_number=preDefCentroids;
			for(int i =0; i<cluster_number;i++)
				CentroidIndexes.offer(i);
		}
		else
		{

			Path centroid_file = new Path(InitialCentroidFolder+CENTROID_FILE_NAME);
			FileSystem f = FileSystem.get(new Configuration());
			DataInputStream centroidStream = new DataInputStream(f.open(centroid_file));
			BufferedReader buff = new BufferedReader(new InputStreamReader(centroidStream));


			while ((line =buff.readLine())!=null) 
			{
				CentroidIndexes.offer(Integer.parseInt(line));
			}

			buff.close();
			cluster_number = CentroidIndexes.size();

		}


		Path input_file = new Path(input+DATA_FILE_NAME);
		FileSystem fs = FileSystem.get(new Configuration());
		DataInputStream stream = new DataInputStream(fs.open(input_file));
		BufferedReader br = new BufferedReader(new InputStreamReader(stream));

		int line_idx =0; 
		int clusterNo = 0;

		while((!CentroidIndexes.isEmpty())  && ((line=br.readLine())!=null))
		{
			if(line_idx != CentroidIndexes.peek())
			{
				line_idx++;
				continue;
			}
			CentroidIndexes.poll();
			String[] sp = line.split("\\t");
			Double[] temp = new Double[sp.length-2];
			for(int j=0;j<sp.length-2;j++)
			{
				temp[j]=Double.parseDouble(sp[j+2]);
			}
			centroid.put(clusterNo++,temp);
			line_idx++;

		}	
		br.close();

		/*
	Path input_file = new Path(input+DATA_FILE_NAME);
	FileSystem fs = FileSystem.get(new Configuration());
	DataInputStream stream = new DataInputStream(fs.open(input_file));
	BufferedReader br = new BufferedReader(new InputStreamReader(stream));
	String line;

	for(int i=0;i< cluster_number;i++)
	{
		line = br.readLine();
		String[] sp = line.split("\\t");
		Double[] temp = new Double[sp.length-2];
		for(int j=0;j<sp.length-2;j++)
		{
			temp[j]=Double.parseDouble(sp[j+2]);
		}
		centroid.put(i,temp);
	}

	br.close();
		 */

		while(true)
		{


			Configuration conf = new Configuration();
			Job job = Job.getInstance(conf, "K-Means");

			job.setJarByClass(KMeans.class);
			job.setMapperClass(MyMapper.class);
			job.setReducerClass(MyReducer.class);

			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);
			FileInputFormat.addInputPath(job, new Path(input+DATA_FILE_NAME));
			FileOutputFormat.setOutputPath(job, new Path(output + String.valueOf(iteration)));
			//System.exit(job.waitForCompletion(true) ? 0 : 1);
			job.waitForCompletion(true);
			if(checkIfHashMapsSame(centroid,previous))
				break;
			previous = new HashMap<Integer, Double[]>();
			copyHashMap(previous, centroid);            
			iteration++;

		}	
		long endTime = System.currentTimeMillis();
		System.out.println("Time: "+((endTime-startTime)/1000)+" seconds");
		calculateJaccard();


	}

	public static void calculateJaccard()
	{
		int size = RowToCentroid.keySet().size();
		int[][] groundLabels= new int[size][size];
		int[][] predLabels= new int[size][size];
		Double same1s = 0.0;
		Double same0s = 0.0;
		Double different = 0.0;
		Double jaccard = 0.0;
		Double rand = 0.0;
		for(Integer x: RowToCentroid.keySet())
		{
			for(Integer y: RowToCentroid.keySet())
			{
				if(RowToCentroid.get(x).equals(RowToCentroid.get(y))){
					predLabels[x][y]=1;
					predLabels[y][x]=1;
				}

			}		
		}
		for(Integer x: RowToGround.keySet())
		{
			for(Integer y: RowToGround.keySet())
			{
				if(RowToGround.get(x).equals(RowToGround.get(y))){
					groundLabels[x][y]=1;
					groundLabels[y][x]=1;
				}

			}		
		}

		for(int i = 0; i<size;i++)
		{
			for(int j=0; j<size; j++)
			{
				if(groundLabels[i][j]==predLabels[i][j]){
					if(groundLabels[i][j]==1)
						same1s+=1.0;
					else
						same0s+=1.0;		
				}
				else
					different+=1.0;
			}	
		}
		jaccard = (same1s/(same1s+different));
		rand = ((same1s+same0s)/(size*size));

		System.out.println("Cluster Assignment: ");
		for(Integer x: RowToCentroid.keySet())
			System.out.print((RowToCentroid.get(x)+1)+", ");

		System.out.println();
		System.out.println("Jaccard: "+jaccard+" Rand: "+rand);
		System.out.println("Number of Iterations: "+iteration);

	}
	public static void copyHashMap(Map<Integer,Double[]> previous, Map<Integer,Double[]> centroid)
	{
		for(Integer c: centroid.keySet())
		{
			previous.put(c, centroid.get(c));
		}
	}

	public static boolean checkIfHashMapsSame(Map<Integer,Double[]> centroid, Map<Integer,Double[]> previous)
	{
		if(previous==null||centroid==null)
			return false;

		for(Integer x : centroid.keySet())
		{
			for(int i = 0; i<centroid.get(x).length;i++){

				if (!centroid.get(x)[i].equals(previous.get(x)[i])){
					return false;
				}
			}
		}

		return true;	
	}
}
