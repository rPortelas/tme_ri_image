package upmc.ri.struct.model;

import upmc.ri.struct.instantiation.IStructInstantiation;

public abstract class LinearStructModel<X, Y> implements IStructModel<X, Y> {
	IStructInstantiation<X, Y> Inst;
	//what about this instantiation method ?
	double[] w;
	
	public abstract IStructInstantiation<X, Y> instantiation();
	
	public double[] getParameters() {
		return w;
	}
	
	public LinearStructModel(int dim_params) {
		this.w = new double[dim_params];
	}
	
	
	
}
