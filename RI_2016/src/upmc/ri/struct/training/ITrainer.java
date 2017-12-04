package upmc.ri.struct.training;

import java.util.List;

import upmc.ri.struct.Evaluator;
import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.model.IStructModel;

public interface ITrainer<X,Y> {
	public void  train(List<STrainingSample<X, Y>> lts , IStructModel<X,Y> model);

	public void train(List<STrainingSample<X, Y>> lts, IStructModel<X,Y> model,
			int epochs_nb, double lr, double regul, Evaluator<X,Y> ev);

}
