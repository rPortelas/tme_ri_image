package upmc.ri.struct.training;
import upmc.ri.struct.*;
import upmc.ri.struct.model.*;
import upmc.ri.struct.instantiation.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Set;

import upmc.ri.utils.*;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.model.IStructModel;
import java.util.Collections;

public class SGDTrainer<X, Y> implements ITrainer<X, Y> {

	//public static final int graph_step = 200;
	Evaluator<X, Y> Ev;
	
	double eta; //should we choose it here ?
	double lambda; //same
	int T; //epoch number
	int N; //data size
	double[] w;
	
	
	public void train(List<STrainingSample<X, Y>> lts, IStructModel<X, Y> model, int epochs_nb, double learning_rate, double regul_rate, Evaluator<X,Y> Ev) {
		this.eta = learning_rate;
		this.lambda = regul_rate;
		this.T = epochs_nb;
		this.Ev = Ev;
		
		//get Instantiation object
		IStructInstantiation <X,Y> Inst = model.instantiation();

		//init random obj
		Random rand = new Random();
		
		this.w = model.getParameters();
		this.N = lts.size(); //size of train dataset, maybe ?
		System.out.println(N);

		for(int e = 0; e < this.T; e++) {
			for (int it = 0; it < N; it++) {
				STrainingSample<X, Y> TSample = lts.get(rand.nextInt(N));
				X x_i = TSample.input;
				Y y_i = TSample.output;
				
				//compute prediction
				Y y_hat = model.lai(TSample);

				//compute grad
				double[] grad = VectorOperations.substract(Inst.psi(x_i, y_hat), Inst.psi(x_i, y_i));
				
				//update w
				for (int k = 0; k < this.w.length; k++) {
					this.w[k] = this.w[k] - this.eta * (this.lambda * this.w[k] + grad[k]);
				}
			}
			//Eval model
			//double loss = convex_loss(TSample, y_hat, this.w, Inst, model);
			Ev.evaluate();
			System.out.println("Epoch: " + e);
			System.out.println("Loss on train = " + Ev.getErr_train());
			System.out.println("Loss on test = " + Ev.getErr_test());
		}
	}
	
	//computes convex_loss as defined is equation (1) of tme's pdf
	public double convex_loss(STrainingSample<X, Y> TSample, Y y_hat, double[] w, IStructInstantiation <X,Y> Inst, IStructModel<X, Y> model) {
		Y y_i = TSample.output;
		X x_i = TSample.input;
		
		//regul part
		double regul_part = (this.lambda / 2) * VectorOperations.norm2(w);
		
		//perfomances part
		double perf_part = 0;
		for (int it = 0; it < this.N; it++) {	
			//get all possible y
			Set<Y> y_list = Inst.enumerateY();
			
			//get y giving highest delta + phi score
			double best_score = -1 * Double.MAX_VALUE;
			for (Y y : y_list) {
				double score = Inst.delta(y_i, y) + VectorOperations.dot(Inst.psi(x_i, y), w);
				if (score >= best_score) {
					best_score = score;
				}
			}
			perf_part += best_score - VectorOperations.dot(Inst.psi(x_i, y_i),w);
		}
		perf_part /= this.N;
		
		return regul_part + perf_part;
	}

	public void train(List<STrainingSample<X, Y>> lts, IStructModel<X, Y> model) {
		System.out.println("SHOULD NOT BE CALLED");
	}

}
