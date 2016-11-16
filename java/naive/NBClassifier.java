package naive;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.ml.data.transformation.EventsAdapter;
import org.ml.data.utils.Logger;
import org.ml.domain.Event;
import org.ml.domain.GenericEvent;
import org.ml.domain.GenericEvents;
import org.ml.domain.Outcome;
import org.ml.domain.attributes.AttributeValue;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class NBClassifier  implements EventsClassifier {

	
	Classifier model;
	EventsAdapter adapter = new EventsAdapter();
	List<Event> events = new ArrayList<Event>();
	
	
	@Override
	public Outcome classify(GenericEvent event) {
		ArrayList<Attribute> featureVector = new ArrayList<Attribute>();
		GenericEvents events = new GenericEvents(Arrays.<GenericEvent>asList(event));
		events.getAttributeTypes().stream().forEach(a -> {
			List<String> possibleValues = new ArrayList<String>();
			List<AttributeValue> attributePossibleValues = events.getAttributePossibleValues(a);
			attributePossibleValues.stream().forEach(v -> {
				possibleValues.add(v.name());
			});
			featureVector.add(new Attribute(a.name(), possibleValues));
		});
		
		List<String> classPossibleValues = Arrays.<Outcome>asList(Outcome.values())
				.stream()
				.map(o -> o.name())
				.collect(Collectors.toList());
		Attribute theClass = new Attribute("Class", classPossibleValues);
		featureVector.add(theClass);
		
		try {
			/* test model on different season */
			Instances testingInstances = new Instances("TrainingSet",
					featureVector, featureVector.size());
			testingInstances.setClassIndex(featureVector.size()-1);
			List<Instance> allEventsTesting = getInstances(featureVector, events.getEvents());
			testingInstances.addAll(allEventsTesting);
			
			Evaluation evaluation = new Evaluation(testingInstances);
			double[] probs = evaluation.evaluateModel(model, testingInstances);
			
//			Logger.debug("Evaluation complete");
//			Logger.debug(evaluation.toSummaryString());
			
			if(probs[0]==0){
				return Outcome.Home;
			}else if(probs[0]==1){
				return Outcome.Away;
			}else if(probs[0]==2){
				return Outcome.Draw;
			}
			
		} catch (Exception e) {
			throw new RuntimeException("Learning failed", e);
		}
		return null;
	}
	

	public Evaluation classify(GenericEvents events){
		ArrayList<Attribute> featureVector = new ArrayList<Attribute>();
		events.getAttributeTypes().stream().forEach(a -> {
			List<String> possibleValues = new ArrayList<String>();
			List<AttributeValue> attributePossibleValues = events.getAttributePossibleValues(a);
			attributePossibleValues.stream().forEach(v -> {
				possibleValues.add(v.name());
			});
			featureVector.add(new Attribute(a.name(), possibleValues));
		});
		
		List<String> classPossibleValues = Arrays.<Outcome>asList(Outcome.values())
				.stream()
				.map(o -> o.name())
				.collect(Collectors.toList());
		Attribute theClass = new Attribute("Class", classPossibleValues);
		featureVector.add(theClass);
		
		try {
			/* test model on different season */
			Instances testingInstances = new Instances("TrainingSet",
					featureVector, featureVector.size());
			testingInstances.setClassIndex(featureVector.size()-1);
			List<Instance> allEventsTesting = getInstances(featureVector, events.getEvents());
			testingInstances.addAll(allEventsTesting);
			
			Evaluation evaluation = new Evaluation(testingInstances);
			double[] probs = evaluation.evaluateModel(model, testingInstances);
			
			return evaluation;
			
		} catch (Exception e) {
			throw new RuntimeException("Learning failed", e);
		}
	}
	
	
	@Override
	public void learn(GenericEvents events) {
		ArrayList<Attribute> featureVector = new ArrayList<Attribute>();
		events.getAttributeTypes().stream().forEach(a -> {
			List<String> possibleValues = new ArrayList<String>();
			List<AttributeValue> attributePossibleValues = events.getAttributePossibleValues(a);
			attributePossibleValues.stream().forEach(v -> {
				possibleValues.add(v.name());
			});
			featureVector.add(new Attribute(a.name(), possibleValues));
		});
		
		List<String> classPossibleValues = Arrays.<Outcome>asList(Outcome.values())
				.stream()
				.map(o -> o.name())
				.collect(Collectors.toList());
		Attribute theClass = new Attribute("Class", classPossibleValues);
		featureVector.add(theClass);
		
		/* number of attributes + 1 for class */
		Instances trainingInstances = new Instances("TrainingSet", featureVector, featureVector.size());
		trainingInstances.setClassIndex(featureVector.size()-1); // class will be the last

		List<Instance> allEventsTraining = getInstances(featureVector, events.getEvents());

		Logger.debug("All Events");
		Logger.debug(allEventsTraining);

		trainingInstances.addAll(allEventsTraining);

		NaiveBayes naiveBayes = new NaiveBayes();
		model = (Classifier) naiveBayes;
		try {
			model.buildClassifier(trainingInstances);
			
			/* test model on different season */
			Instances testingInstances = new Instances("TrainingSet",
					featureVector, featureVector.size());
			testingInstances.setClassIndex(featureVector.size()-1);
			List<Instance> allEventsTesting = getInstances(featureVector, events.getEvents());
			testingInstances.addAll(allEventsTesting);
			
			Evaluation evaluation = new Evaluation(testingInstances);
			evaluation.evaluateModel(model, testingInstances);
			
			Logger.debug(evaluation.toSummaryString());
			
			Logger.debug(naiveBayes.toString());
		} catch (Exception e) {
			throw new RuntimeException("Learning failed", e);
		}

	}		
	


	public void printStatistics(){
		try {
			Evaluation evaluation = classify(adapter.toGenericEvents(events));
			Logger.info("Evaluation complete");
			Logger.info(evaluation.toSummaryString());
			Logger.info(evaluation.toMatrixString("Confusion Matrix"));
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}

	private List<Instance> getInstances(ArrayList<Attribute> featureVector, List<GenericEvent> events) {
		printInTabular(events);
		List<Instance> instances = new ArrayList<Instance>();
		events.stream().forEach(e -> {
			List<String> values = new ArrayList<String>();

			e.getAttributes().stream().forEach(a -> {
				String value = a.getValue().name();
				values.add(value);
			});
			values.add(e.getTheClass().name());
			Instance instance = createDenseInstance(3, featureVector, values.toArray(new String[values.size()]));
			instances.add(instance);
		});
		return instances;
	}


	private void printInTabular(List<GenericEvent> cleanEvents) {
		Logger.debug("Table:");
		cleanEvents.stream().forEach(e -> {
			Logger.debug(e + " ");
			e.getAttributes().stream().forEach(a -> {
				Logger.debug(a + " = " + e.getTheClass());
			});
		});
	}

	
	private DenseInstance createDenseInstance(int size, List<Attribute> featureVector, String... values) {
		assert featureVector.size() == values.length;
		DenseInstance instance = new DenseInstance(featureVector.size());
		for (int i = 0; i < values.length; i++) {
			Logger.debug(featureVector + " " + values[i]);
			instance.setValue(featureVector.get(i), values[i]);
		}
		return instance;
	}
	

	
}
