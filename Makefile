install:
	pip install -r requirements.txt
clean:
	find ./|grep '\.pyc'|xargs rm
	find ./|grep '\.pyo'|xargs rm
	find ./|grep '\.tstc'|xargs rm
	find ./|grep '\.tsto'|xargs rm
