package upmc.ri.struct.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import upmc.ri.struct.STrainingSample;
import upmc.ri.struct.instantiation.IStructInstantiation;
import upmc.ri.struct.ranking.RankingFunctions;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.utils.VectorOperations;

public class RankingStructModel extends LinearStructModel<List<double[]>,RankingOutput> {

	public RankingStructModel(int dim_params, IStructInstantiation<List<double[]>, RankingOutput> Inst) {
		super(dim_params, Inst);
		
	}

	@Override
	public RankingOutput predict(STrainingSample<List<double[]>, RankingOutput> ts) {
		List<Double> scores = new ArrayList<Double>(ts.input.size());
		
		//computes scores for each image, using our model's parameters w
		for (double[] bow : ts.input) {
			scores.add(VectorOperations.dot(this.w, bow));
		}
		
		//now order images's indices by image score
		int[] sorted_index = IntStream.range(0,scores.size()).boxed().sorted((i,j) -> (scores.get(i) <= scores.get(j)?1:-1)).mapToInt(ele -> ele).toArray();
		
		//creating ranking list for ranking output object
		List<Integer> ranking = new ArrayList<Integer>(sorted_index.length);
		for(int i=0;i<sorted_index.length;i++) {
			ranking.add(sorted_index[i]);
		}
		return new RankingOutput(ts.output.getNbPlus(),ranking, ts.output.getLabelsGT());
	}

	@Override
	public RankingOutput lai(STrainingSample<List<double[]>, RankingOutput> ts) {
		return RankingFunctions.loss_augmented_inference(ts, this.getParameters());
	}

	@Override
	public void setInst(IStructInstantiation<List<double[]>, RankingOutput> Inst) {
		this.Inst = Inst;
		
	}
	
	@Override
	public void setParameters(double[] params) {
		this.w = params;
	}	
	

}
