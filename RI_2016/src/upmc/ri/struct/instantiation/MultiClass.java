package upmc.ri.struct.instantiation;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import static java.lang.Math.toIntExact;

import java.math.BigDecimal;
import java.math.RoundingMode;

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

		this.map = new HashMap<String,Integer>();
		int y_ind = 0;
		for (String y : this.set) {
			map.put(y, y_ind);
			y_ind++;
		}
	}
	
	public double[][] confusionMatrix(List<String> y_hat_list,List<String> y_list) {
		int class_nb = this.set.size();
		double[][] mat_conf = new double[class_nb][class_nb];
		Map<String, Long> counts = y_list.stream().collect(Collectors.groupingBy(e -> e, Collectors.counting()));
	
		//String[] classes = this.set.toArray(new String[class_nb]);
		String[] classes = {"taxi","ambulance","minivan","acoustic_guitar","electric_guitar","harp","wood-frog","tree-frog","european_fire_salamander"};
		
		//For each class i
		for(int i=0;i<class_nb;i++) {
			int index = y_list.indexOf(classes[i]);
			int len = toIntExact(counts.get(classes[i]));
			
			//For each class j
			for(int j=0;j<class_nb;j++) {
				double nb_j_predictions = Collections.frequency(y_hat_list.subList(index, index + len), classes[j]);
				mat_conf[i][j] = round(nb_j_predictions / len,3);
				
				
			}
		}
		System.out.println(Arrays.toString(classes));
		System.out.println(Arrays.deepToString(mat_conf).replace("], ","]\n"));
		return mat_conf;
	}

	public double[] psi(double[] x, String y) {
		int input_dim = x.length;
		
		//init psi
		double[] psi = new double[this.set.size()*input_dim];
		
		//now add x into y's part of psi
		int y_ind = this.map.get(y);
		
		for(int i = y_ind * input_dim; i < (y_ind * input_dim) + input_dim; i++) {
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
	
	public static double round(double value, int places) {
	    if (places < 0) throw new IllegalArgumentException();

	    BigDecimal bd = new BigDecimal(value);
	    bd = bd.setScale(places, RoundingMode.HALF_UP);
	    return bd.doubleValue();
	}
}
