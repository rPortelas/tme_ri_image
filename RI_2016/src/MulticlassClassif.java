import java.util.ArrayList;
import java.util.List;

import upmc.ri.struct.DataSet;
import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;
import upmc.ri.struct.instantiation.MultiClass;
import upmc.ri.struct.instantiation.MultiClassHier;
import upmc.ri.struct.model.IStructModel;
import upmc.ri.struct.model.LinearStructModel_Ex;
import upmc.ri.struct.training.ITrainer;
import upmc.ri.struct.training.SGDTrainer;

public class MulticlassClassif {
	
	public static int input_dim = 250;
	public static double regul = 10e-6;
	public static double lr = 10e-2;
	public static int epochs_nb = 100;
		

	public static void main(String[] args) {
		System.out.println("load data & run PCA");
		DataSet<double[], String> dSet = VisualIndexes.buildDataset(input_dim);
		
		System.out.println("init MultiClass Instantiation");
		//-----------------------------------------------------------------------------------------
		//IStructInstantiation<double[],String> Inst = new MultiClass();
		IStructInstantiation<double[],String> Inst = new MultiClassHier();
		
		System.out.println("init model");
		IStructModel<double[],String> model = new LinearStructModel_Ex<double[], String>(input_dim,Inst);
		
		System.out.println("create Evaluator");
		Evaluator<double[],String> Ev = new Evaluator<double[], String>();
		Ev.setListtrain(dSet.listtrain);
		Ev.setListtest(dSet.listtest);
		Ev.setModel(model);
		
		System.out.println("train model");
		ITrainer<double[],String> Trainer = new SGDTrainer<double[],String>();
		Trainer.train(dSet.listtrain,model,epochs_nb,lr,regul,Ev);
		
		System.out.println("Compute confusion matrix");
		Ev.evaluate();
		MultiClass MC = (MultiClass) Inst;
		
		List<String> test_y = new ArrayList<String>(dSet.listtest.size());
		for (STrainingSample<double[], String> test_sample : dSet.listtest) {
			test_y.add(test_sample.output);
		}
		
		//SHOULD RETURN 1
		//double[][] mat_conf = MC.confusionMatrix(test_y,test_y);
		
		double[][] mat_conf = MC.confusionMatrix(Ev.getPred_test(),test_y);
		

	}

}
