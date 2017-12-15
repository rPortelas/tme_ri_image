import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
		DataSet<List<double[]>, RankingOutput> dSet = RankingFunctions.convertClassif2Ranking(classif_dSet, "taxi");
		
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
		
		System.out.println("train model");
		ITrainer<List<double[]>, RankingOutput> Trainer = new SGDTrainer<List<double[]>, RankingOutput>();
		Trainer.train(dSet.listtrain,model,epochs_nb,lr,regul,Ev);
		
		RankingOutput y_train_pred = model.predict(dSet.listtrain.get(0));
		double[][] rp = RankingFunctions.recalPrecisionCurve(y_train_pred);
		Drawing.traceRecallPrecisionCurve(dSet.listtrain.get(0).output.getNbPlus(), rp);
		/*
		//System.out.println("Compute confusion matrix");
		Ev.evaluate();
		MultiClass MC = (MultiClass) Inst;
			
		
		
		List<String> test_y = new ArrayList<String>(dSet.listtest.size());
		for (STrainingSample<double[], String> test_sample : dSet.listtest) {
			test_y.add(test_sample.output);
		}
		
		double[][] mat_conf = MC.confusionMatrix(Ev.getPred_test(),test_y);
		
		//get mean error on test using 0-1 and hierarchical models
		IStructInstantiation<double[],String> Inst_test_hier = new MultiClassHier();
		IStructInstantiation<double[],String> Inst_test_01 = new MultiClass();
		
		if(use_hier_model) {
			System.out.println("Using a SVM trained with hierarchical loss: ");
		}
		else {
			System.out.println("Using a SVM trained with 0-1 loss: ");
		}
		
		model.setInst(Inst_test_hier);
		Ev.setModel(model);
		Ev.evaluate();
		System.out.println("Mean error on test set using hierarchical loss = " + Ev.getErr_test());
		
		model.setInst(Inst_test_01);
		Ev.setModel(model);
		Ev.evaluate();
		System.out.println("Mean error on test set using 0-1 loss = " + Ev.getErr_test());
		*/

	}

}
