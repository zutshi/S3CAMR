install:
	pip install -r requirements.txt

checkpy3compat:
	pylint --py3k ./src/*|less
clean:
	find ./|grep '\.pyc'|xargs rm -f
	find ./|grep '\.pyo'|xargs rm -f
	find ./|grep '\.tstc'|xargs rm -f
	find ./|grep '\.tsto'|xargs rm -f
