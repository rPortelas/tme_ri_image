package upmc.ri.struct.model;

import upmc.ri.struct.instantiation.IStructInstantiation;

public abstract class LinearStructModel<X, Y> implements IStructModel<X, Y> {
	IStructInstantiation<X, Y> SInst;
	//what about this instantiation method ?
	double[] params;
	
	public double[] getParameters() {
		return params;
	}
	
	public LinearStructModel(int dim_params) {
		//what should i do ?
	}
	
	
	
}
