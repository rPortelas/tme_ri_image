package upmc.ri.struct.instantiation;

import java.util.Arrays;

import edu.cmu.lti.lexical_db.ILexicalDatabase;
import edu.cmu.lti.lexical_db.NictWordNet;
import edu.cmu.lti.ws4j.RelatednessCalculator;
import edu.cmu.lti.ws4j.impl.WuPalmer;

public class MultiClassHier extends MultiClass{
	
	double[][] similarities;
	public static final String[] ordered_classes = {"taxi","ambulance","minivan","acoustic_guitar","electric_guitar","harp","wood-frog","tree-frog","european_fire_salamander"};
	
	public MultiClassHier() {
		super();

		//now computes distance matrix using WordNet
		RelatednessCalculator rc;
		ILexicalDatabase db = new NictWordNet();
		rc = new WuPalmer(db);
		//this.similarities = rc.getSimilarityMatrix(ordered_classes, ordered_classes);
		this.similarities = rc.getNormalizedSimilarityMatrix(ordered_classes, ordered_classes);
		System.out.println(Arrays.deepToString(similarities).replace("], ","]\n"));
	}
	
	//implements Hierarchical loss
	public double delta(String y1, String y2) {
		if (y1.compareTo(y2) == 0){
			//y1 == y2
			return 0;
		}
		return (1 - this.similarities[this.map.get(y1)][this.map.get(y2)]);
	}

}
