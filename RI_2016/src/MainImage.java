import java.util.Set;
import java.util.List;
import upmc.ri.io.*;
import upmc.ri.index.*;
import upmc.ri.struct.*;

public class MainImage {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Set<String> labels = ImageNetParser.classesImageNet();			
		
		List <STrainingSample<double[],String>> listTrain ;
		List <STrainingSample<double[],String>> listTest ;
		
		String path_to_bows = "/home/kalifou/Documents/m2/ri/tmes/tme_ri_image/sbow/";
		// iterate over set of labels
		for( String l : labels){
			List <ImageFeatures> F ;
			try {
				F = ImageNetParser.getFeatures(path_to_bows+ l +".txt");
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			int cpt=0;
			for( ImageFeatures ft : F){
				
				double[] bow = VIndexFactory.computeBow(ft);
				if(cpt<800){
					
					listTrain.add(STrainingSample<bow,ft>);
				}
				else{
					listTest.add(STrainingSample<bow,ft> );
				}
				cpt+=1;
				
			}
			
			
		}
		
	}

}
