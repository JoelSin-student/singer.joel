- raw_data:
	- Awinda: MVN export files in xlsx format
	- Insoles: OpenGo export text files in tab-separated format

- clean_data:
	all preprocessed data: .txt being the insoles' (effectively in csv after preprocessing), .csv being the Awinda's

- training_data:
	data that serves as training model (~80% of clean_data)
	- Awinda: MVN preprocessed files in xlsx format
	- Insoles: OpenGo preprocessed text files in csv format

- test_data:
	data that serves as testing model prediction after training (~20% of clean_data)
	- Awinda: MVN preprocessed files in xlsx format
		- Other testing files: we can only predict one file at a time. In this folder are the other testing files
	- Insoles: OpenGo preprocessed text files in csv format
		- Other testing files: we can only predict one file at a time. In this folder are the other testing files