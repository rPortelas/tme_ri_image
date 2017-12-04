package upmc.ri.utils;

public class VectorOperations {

	public static double dot(double [] v1 , double [] v2){
		double res = 0.0;
		for(int i=0;i<v1.length;i++){
			res+= v1[i]*v2[i];
		}
		return res;
	}
	
	public static double norm2(double[]v){
		return dot(v,v);
	}
	public static double norm(double[]v){
		return Math.sqrt(norm2(v));
	}
	
	public static double[] substract(double[] a1, double[] a2) {
		assert(a1.length == a2.length);
		int size = a1.length;
		double[] res = a1.clone(); 
		for (int i=0;i<size;i++){
			res[i] -= a2[i];
		}
		return res;
	}
}
