package upmc.ri.struct.model;

import java.util.Set;

import upmc.ri.struct.instantiation.IStructInstantiation;

public abstract class LinearStructModel<X, Y> implements IStructModel<X, Y> {
	IStructInstantiation<X, Y> Inst;
	double[] w;

	public IStructInstantiation<X, Y> instantiation(){
		assert(Inst!=null);
		return this.Inst;
	}

	public double[] getParameters() {
		return w;
	}

	public LinearStructModel(int dim_params,IStructInstantiation<X, Y> Inst ) {
		Set<Y> ySet = Inst.enumerateY();
		int ySet_size;
		if (ySet == null) {
			ySet_size = 1;
		}
		else {
			ySet_size = ySet.size();
		}		
		this.w = new double[dim_params * ySet_size];
		this.Inst = Inst;
	}

}
