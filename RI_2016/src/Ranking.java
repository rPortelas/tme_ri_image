import java.awt.Panel;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import upmc.ri.struct.DataSet;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;
import upmc.ri.struct.instantiation.RankingInstantiation;
import upmc.ri.struct.model.IStructModel;
import upmc.ri.struct.model.RankingStructModel;
import upmc.ri.struct.ranking.RankingFunctions;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.struct.training.ITrainer;
import upmc.ri.struct.training.SGDTrainer;
import upmc.ri.utils.Drawing;

public class Ranking {
	
	public static int input_dim = 250;
	public static double regul = 10e-6;
	public static double lr = 10;
	public static int epochs_nb = 50;
	
	// !!!!!! CHOOSE THE LIST OF QUERIES YOU WANT TO RUN HERE !!!!!!
	public static String[] queries = {"taxi","ambulance","minivan","acoustic_guitar","electric_guitar","harp","wood-frog","tree-frog","european_fire_salamander"};
	
	// !!!!!! SET UP ABSOLUTE FILE PATH TO SBOW FOLDER
	public static String path_to_bows = "/home/rportelas/Documents/RI/tme_ri_image/sbow/";
	
	public static void main(String[] args) {

		System.out.println("load data & run PCA");
		DataSet<double[], String> classif_dSet = VisualIndexes.buildDataset(input_dim,path_to_bows);
		
		List<Double> all_AP_train = new ArrayList<Double>(queries.length);
		List<Double> all_AP_test = new ArrayList<Double>(queries.length);

		//Train a SSVM for ranking for each query
		for(String query : queries) {
			System.out.println("\n");
			System.out.println("QUERY : " + query);
			System.out.println("\n");
			//Convert to ranking data
			DataSet<List<double[]>, RankingOutput> dSet = RankingFunctions.convertClassif2Ranking(classif_dSet, query);

			System.out.println("init Ranking Instantiation");

			IStructInstantiation<List<double[]>, RankingOutput> Inst;
			Inst = new RankingInstantiation();

			System.out.println("init model");
			IStructModel<List<double[]>, RankingOutput> model = new RankingStructModel(input_dim, Inst);

			System.out.println("create Evaluator");
			Evaluator<List<double[]>, RankingOutput> Ev = new Evaluator<List<double[]>, RankingOutput>();
			Ev.setListtrain(dSet.listtrain);
			Ev.setListtest(dSet.listtest);
			Ev.setModel(model);

			System.out.println("train model during " + epochs_nb + " epochs...");
			ITrainer<List<double[]>, RankingOutput> Trainer = new SGDTrainer<List<double[]>, RankingOutput>();
			Trainer.train(dSet.listtrain,model,epochs_nb,lr,regul,Ev);

			//Evaluate prediction on train
			RankingOutput y_train_pred = model.predict(dSet.listtrain.get(0));
			
			double AP_train = RankingFunctions.averagePrecision(y_train_pred);
			all_AP_train.add(AP_train);
			System.out.println("Query: " + query + ",train Average Precision: " + AP_train);
			//System.out.println(RankingFunctions.averagePrecision(y_train_pred));

			double[][] rp = RankingFunctions.recalPrecisionCurve(y_train_pred);
			BufferedImage bIm = Drawing.traceRecallPrecisionCurve(dSet.listtrain.get(0).output.getNbPlus(), rp);

			JFrame frame = new JFrame("Query: " + query + ", Courbe Recall-Precision en Train");
			Panel panel = new ImageDisplay(bIm);
			frame.getContentPane().add(panel);
			frame.setSize(500, 500);
			frame.setVisible(true);

			//Evaluate prediction on test
			RankingOutput y_test_pred = model.predict(dSet.listtest.get(0));
			
			double AP_test = RankingFunctions.averagePrecision(y_test_pred);
			all_AP_test.add(AP_test);
			System.out.println("Query: " + query + ",test Average Precision: " + RankingFunctions.averagePrecision(y_test_pred));
			//System.out.println(RankingFunctions.averagePrecision(y_test_pred));

			double[][] rp_test = RankingFunctions.recalPrecisionCurve(y_test_pred);
			BufferedImage bIm_test = Drawing.traceRecallPrecisionCurve(dSet.listtest.get(0).output.getNbPlus(), rp_test);

			JFrame frame_test = new JFrame("Query: " + query + ", Courbe Recall-Precision en Test");
			Panel panel_test = new ImageDisplay(bIm_test);
			frame_test.getContentPane().add(panel_test);
			frame_test.setSize(500, 500);
			frame_test.setVisible(true);
			
			//save RP plots
			File train_rp = new File("Train-Recall-Prec-" + query);
			File test_rp = new File("Test-Recall-Prec-" + query);
			try {
				ImageIO.write(bIm, "png", train_rp);
				ImageIO.write(bIm_test, "png", test_rp);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			

		}
		
		double mean_AP_test = 0;
		double mean_AP_train = 0;
		//Compute mean AP
		for(int i=0;i<all_AP_test.size();i++) {
			mean_AP_test = mean_AP_test + all_AP_test.get(i);
			mean_AP_train = mean_AP_train + all_AP_train.get(i);
		}
		mean_AP_test = mean_AP_test / all_AP_test.size();
		mean_AP_train = mean_AP_train / all_AP_train.size();
		
		System.out.println("Mean AP pour la liste de queries suivante: " + Arrays.toString(queries));
		System.out.println("moyenne train AP = " + mean_AP_train + ", moyenne test AP" + mean_AP_test);
		
		
	}

}
