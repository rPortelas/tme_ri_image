package upmc.ri.struct.training;
import upmc.ri.struct.*;
import upmc.ri.struct.model.*;
import upmc.ri.struct.instantiation.*;
import java.util.List;
import java.util.Random;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.model.IStructModel;

public class SGDTrainer<X, Y> implements ITrainer<X, Y> {

	Evaluator<X, Y> Ev;
	
	public void train(List<STrainingSample<X, Y>> lts, IStructModel<X, Y> model) {
		//init random obj
		Random rand = new Random();
		
		//should we init them here or are they already init ?
		double[] w = model.getParameters();
		double eta = -1; //should we choose it here ?
		double lambda = -1; //same
		
		//T = number of epochs, should we choose it here ?
		int T = 1; //???
		int N = lts.size(); //size of train dataset, maybe ?
		
		for(int e = 0; e < T; e++) {
			for (int it = 0; it < N; it++) {
				STrainingSample<X, Y> TSample = lts.get(rand.nextInt(N + 1));
				Y pred = lai(TSample); //wtf you say bru ?
				//why should we define convex_loss if we have lai
				
				//compute grad
				double[] grad = psi(TSample.input,TSample.output);
				
				//update w
				for (int k = 0; k < w.length; k++) {
					w[k] = w[k] - eta * (lambda * w[k] + grad[k]);
				}		
			}
		}
		return w;
		
	}
	
	
	//return double or int or array ????
	//computes convex_loss as defined is equation (1) of tme's pdf
	public double convex_loss(double[] w) {
		return -1;
	}

}
