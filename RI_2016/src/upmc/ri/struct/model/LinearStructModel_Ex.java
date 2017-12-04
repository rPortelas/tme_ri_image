package upmc.ri.struct.model;

import java.util.Set;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;
import upmc.ri.utils.VectorOperations;

public class LinearStructModel_Ex<X, Y> extends LinearStructModel<X, Y> {

	public LinearStructModel_Ex(int dim_params, IStructInstantiation<X, Y> Inst) {
		super(dim_params, Inst);
	}

	@Override
	public Y predict(STrainingSample<X, Y> TSample) {
		X x_i = TSample.input;
		
		Set<Y> y_list = this.Inst.enumerateY();
		
		//get y giving highest delta + phi score
		double best_score = -1 * Double.MAX_VALUE;
		Y best_y = null;
		for (Y y : y_list) {
			double score = VectorOperations.dot(Inst.psi(x_i, y), w);
			if (score >= best_score) {
				best_score = score;
				best_y = y;
			}
		}
		assert(best_score != (-1 * Double.MAX_VALUE));
		return best_y;
	}

	@Override
	public Y lai(STrainingSample<X, Y> TSample) {
		Y y_i = TSample.output;
		X x_i = TSample.input;
		
		Set<Y> y_list = this.Inst.enumerateY();
		
		//get y giving highest delta + phi score
		double best_score = -1 * Double.MAX_VALUE;
		Y best_y = null;
		for (Y y : y_list) {
			double score = Inst.delta(y, y_i) + VectorOperations.dot(Inst.psi(x_i, y), w);
			if (score >= best_score) {
				best_score = score;
				best_y = y;
			}
		}
		assert(best_score != (-1 * Double.MAX_VALUE));
		return best_y;
	}	
}
