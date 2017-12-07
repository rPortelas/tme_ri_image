package upmc.ri.struct.instantiation;

import java.util.List;
import java.util.Set;

import upmc.ri.struct.ranking.RankingOutput;
import upmc.ri.struct.ranking.*;

public class RankingInstantiation implements IStructInstantiation<List<double[]>, RankingOutput> {

	@Override
	public double[] psi(List<double[]> x, RankingOutput y) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double delta(RankingOutput y1, RankingOutput y2) {
		//The fuck why aren't we using y1 ?
		return 1 - RankingFunctions.averagePrecision(y2);
	}

	@Override
	public Set<RankingOutput> enumerateY() {
		//will not be used for ranking outputs
		return null;
	}

}
