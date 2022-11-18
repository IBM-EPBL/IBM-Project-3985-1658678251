#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Installing amchine learning service
get_ipython().system('pip install watson-machine-learning-client')


# In[17]:


#Replace the credentials that you got from Watson Machine Learning Service
from ibm_watson_machine_learning import APIClient
wml_credentials={
    "url":"https://us-south.ml.cloud.ibm.com",
    "apikey":"VbAtsHDnOMMnU006loi4zVpje-qQj2HZvnwWv_QLlhPP"
}
client=APIClient(wml_credentials)


# In[18]:


client=APIClient(wml_credentials)


# In[19]:


client


# In[20]:


def guid_from_space_name(client,space_name):
    space=client.spaces.get_details()
    #print(space)
    return(next(item for item in space['resources'] if item['entity']['name']==space_name)['metadata']['id'])


# In[22]:


space_uid=guid_from_space_name(client, 'project-deployments')
print("space UID="+space_uid)


# In[23]:


client.set.default_space(space_uid)


# In[24]:


client.software_specifications.list()


# In[25]:


software_spec_uid = client.software_specifications.get_uid_by_name("tensorflow_rt22.1-py3.9")
software_spec_uid


# In[26]:


model_details = client.repository.store_model(model='my_model_new.tgz',meta_props={
    client.repository.ModelMetaNames.NAME:"CNN",
    client.repository.ModelMetaNames.TYPE:"tensorflow_2.7",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_spec_uid
})


# In[27]:


model_id=client.repository.get_model_uid(model_details)


# In[28]:


model_id


# In[29]:


client.repository.download(model_id,'model.tar.gz')


# In[ ]:




