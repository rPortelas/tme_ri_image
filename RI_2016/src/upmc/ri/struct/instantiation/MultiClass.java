package upmc.ri.struct.instantiation;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class MultiClass implements IStructInstantiation<double[], String> {
	Set<String> set;
	Map<String,Integer> map;

	public MultiClass() {
		this.set = new HashSet<String>();
		this.set.add("taxi");
		this.set.add("ambulance");
		this.set.add("minivan");
		this.set.add("acoustic_guitar");
		this.set.add("electric_guitar");
		this.set.add("harp");
		this.set.add("wood-frog");
		this.set.add("tree-frog");
		this.set.add("european_fire_salamander");
		
		int cpt = 0;
		for (String y : this.set)
			map.put(y, cpt++);
	}

	public double[] psi(double[] x, String y) {
		int input_dim = x.length;
		
		//init psi
		double[] psi = new double[this.set.size()*input_dim];
		
		//now add x into y's part of psi
		int y_ind = this.map.get(y);
		
		for(int i = y_ind * input_dim; i < psi.length; i++) {
			psi[i] = x[i - (y_ind * input_dim)];
		}
		return psi;
	}

	//implements 0-1 loss
	public double delta(String y1, String y2) {
		if (y1.compareTo(y2) == 0){
			//y1 == y2
			return 0;
		}
		return 1;
	}

	public Set<String> enumerateY() {
		return this.set;
	}
}
