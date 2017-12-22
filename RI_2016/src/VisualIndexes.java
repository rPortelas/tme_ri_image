import java.util.Set;
import java.util.Iterator;
import java.util.List;
import upmc.ri.io.*;
import upmc.ri.index.*;
import upmc.ri.struct.*;
import upmc.ri.utils.*;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class VisualIndexes {

	public static DataSet<double[], String> buildDataset(int pca_dim, String path_to_bows) {
		// TODO Auto-generated method stub
		Set<String> labels = ImageNetParser.classesImageNet();
		 
		List <STrainingSample<double[],String>> listTrain = new ArrayList<STrainingSample<double[],String>>();
		List <STrainingSample<double[],String>> listTest = new ArrayList<STrainingSample<double[],String>>();
		
		// iterate over set of labels
		Iterator<String> it = labels.iterator();
		while (it.hasNext()) {
			String label = it.next();
			
			List <ImageFeatures> F = new ArrayList<ImageFeatures>();
			System.out.println(label);
			try {
				F = ImageNetParser.getFeatures(path_to_bows+ label +".txt");
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			//iterate over images to construct train and test lists
			int cpt=0;
			Iterator<ImageFeatures> image_it = F.iterator();
			while (image_it.hasNext()) {
				ImageFeatures ft = image_it.next();
				
				//get normalized bag of words
				double[] bow = VIndexFactory.computeBow(ft);
				STrainingSample<double[],String> data_sample = new STrainingSample<double[], String>(bow, label);
				if(cpt<800){	
					listTrain.add(data_sample);
				}
				else{
					listTest.add(data_sample);
				}
				cpt+=1;
			}
		}
		//create dataset then run PCA on it to reduce sample dimension to pca_dim components
		DataSet<double[],String> dSet = new DataSet<double[],String>(listTrain, listTest);
		
		return PCA.computePCA(dSet , pca_dim);	
	}

}
