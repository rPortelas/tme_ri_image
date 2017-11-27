import java.util.Set;
import java.util.Iterator;
import java.util.List;
import upmc.ri.io.*;
import upmc.ri.index.*;
import upmc.ri.struct.*;
import java.util.ArrayList;

public class MainImage {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Set<String> labels = ImageNetParser.classesImageNet();			
		
		List <STrainingSample<double[],String>> listTrain = new ArrayList<STrainingSample<double[],String>>();
		List <STrainingSample<double[],String>> listTest = new ArrayList<STrainingSample<double[],String>>();
		
		//String path_to_bows = "/home/kalifou/Documents/m2/ri/tmes/tme_ri_image/sbow/";
		String path_to_bows = "/home/rportelas/Documents/RI/tme_ri_image/sbow/";
		
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
			
			int cpt=0;
			Iterator<ImageFeatures> image_it = F.iterator();
			while (image_it.hasNext()) {
				ImageFeatures ft = image_it.next();
				
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
		
	}

}
