package upmc.ri.struct.instantiation;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import edu.cmu.lti.lexical_db.ILexicalDatabase;
import edu.cmu.lti.lexical_db.NictWordNet;
import edu.cmu.lti.ws4j.RelatednessCalculator;
import edu.cmu.lti.ws4j.impl.WuPalmer;
import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.utils.VectorOperations;
import upmc.ri.struct.ranking.*;

public class RankingInstantiation implements IStructInstantiation<List<double[]>, RankingOutput> {

	public RankingInstantiation() {
		super();
	}
	
	public double[] psi(List<double[]> x, RankingOutput y) {
		double[] ret = new double[x.get(0).length];

		List<Integer> ranking = y.getRanking();
		List<Integer> label = y.getLabelsGT();
		for(int i=0;i<ranking.size();i++) {
			//System.out.println(ranking.get(i).);
			//if pos example then compare to all neg examples, else goto next pos example
			if(label.get(ranking.get(i)) == -1) { // neg example
				continue;
			}
			else {
				for(int j=0;j<ranking.size();j++) {
					if(label.get(ranking.get(j)) == 1) { //pos example
						continue;
					}
					else { //if neg example, add bow add to result
						if(i<j) {// +1 factor, positive example is correctly ranked before negative one
							ret = VectorOperations.add(ret, VectorOperations.substract(x.get(ranking.get(i)), x.get(ranking.get(j))));
						}
						else if(i == j){
							System.out.println("ERROR, cannot be positive and negative !");
						}
						else { // i>j : -1 factor, positive example is wrongly ranked after negative one
							double[] val = VectorOperations.substract(x.get(ranking.get(i)), x.get(ranking.get(j)));
							for(int k=0;k<val.length;k++) {
								val[k] = -1 * val[k];
							}
							ret = VectorOperations.add(ret,val);
						}
					}
					
					
				}
			}

		}
		return ret;
	}

	@Override
	public double delta(RankingOutput y1, RankingOutput y2) {
		return 1 - RankingFunctions.averagePrecision(y2);
	}

	@Override
	public Set<RankingOutput> enumerateY() {
		//will not be used for ranking outputs
		return null;
	}

}
