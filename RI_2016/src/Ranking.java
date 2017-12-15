import java.awt.Panel;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
	
	
	public static void main(String[] args) {
		System.out.println("load data & run PCA");
		DataSet<double[], String> classif_dSet = VisualIndexes.buildDataset(input_dim);
		//Convert to ranking data
		DataSet<List<double[]>, RankingOutput> dSet = RankingFunctions.convertClassif2Ranking(classif_dSet, "ambulance");
		
		System.out.println("init Ranking Instantiation");
		
		IStructInstantiation<List<double[]>, RankingOutput> Inst;
		Inst = new RankingInstantiation();
		System.out.println(Inst);
		System.out.println("init model");
		IStructModel<List<double[]>, RankingOutput> model = new RankingStructModel(input_dim, Inst);
		System.out.println(model);
		System.out.println("create Evaluator");
		Evaluator<List<double[]>, RankingOutput> Ev = new Evaluator<List<double[]>, RankingOutput>();
		Ev.setListtrain(dSet.listtrain);
		Ev.setListtest(dSet.listtest);
		Ev.setModel(model);
		
		System.out.println("train model");
		ITrainer<List<double[]>, RankingOutput> Trainer = new SGDTrainer<List<double[]>, RankingOutput>();
		Trainer.train(dSet.listtrain,model,epochs_nb,lr,regul,Ev);
		
		//Evaluate prediction on train
		RankingOutput y_train_pred = model.predict(dSet.listtrain.get(0));
		
		System.out.println("train Average Precision: ");
		System.out.println(RankingFunctions.averagePrecision(y_train_pred));
		
		double[][] rp = RankingFunctions.recalPrecisionCurve(y_train_pred);
		BufferedImage bIm = Drawing.traceRecallPrecisionCurve(dSet.listtrain.get(0).output.getNbPlus(), rp);
		
		JFrame frame = new JFrame("Courbe Recall-Precision en Train");
		Panel panel = new ImageDisplay(bIm);
		frame.getContentPane().add(panel);
		frame.setSize(500, 500);
		frame.setVisible(true);
		
		//Evaluate prediction on test
		RankingOutput y_test_pred = model.predict(dSet.listtest.get(0));
		
		System.out.println("test Average Precision: ");
		System.out.println(RankingFunctions.averagePrecision(y_test_pred));
		
		double[][] rp_test = RankingFunctions.recalPrecisionCurve(y_test_pred);
		BufferedImage bIm_test = Drawing.traceRecallPrecisionCurve(dSet.listtest.get(0).output.getNbPlus(), rp_test);
		
		JFrame frame_test = new JFrame("Courbe Recall-Precision en Test");
		Panel panel_test = new ImageDisplay(bIm_test);
		frame_test.getContentPane().add(panel_test);
		frame_test.setSize(500, 500);
		frame_test.setVisible(true);

	}

}
