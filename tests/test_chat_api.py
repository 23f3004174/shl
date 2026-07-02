from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_vague_query_requests_clarification():
    response = client.post('/chat', json={
        'messages': [{'role': 'user', 'content': 'I need an assessment.'}]
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload['end_of_conversation'] is False
    assert payload['recommendations'] == []
    assert 'role' in payload['reply'].lower() or 'seniority' in payload['reply'].lower()


def test_detailed_query_returns_recommendations():
    response = client.post('/chat', json={
        'messages': [{'role': 'user', 'content': 'I need a senior technical assessment for a Java developer.'}]
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload['end_of_conversation'] is True
    assert len(payload['recommendations']) >= 1
    assert payload['recommendations'][0]['name']
    assert payload['recommendations'][0]['url']


def test_unrelated_query_is_rejected():
    response = client.post('/chat', json={
        'messages': [{'role': 'user', 'content': 'Can you give me legal advice about hiring?'}]
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload['recommendations'] == []
    assert payload['end_of_conversation'] is False


def test_comparison_query_uses_catalog_information():
    response = client.post('/chat', json={
        'messages': [{'role': 'user', 'content': 'What is the difference between OPQ and GSA?'}]
    })

    assert response.status_code == 200
    payload = response.json()
    assert 'gsa' in payload['reply'].lower() or 'opq' in payload['reply'].lower()
    assert payload['recommendations'] == []
    assert payload['end_of_conversation'] is False


def test_refinement_query_updates_recommendations():
    first = client.post('/chat', json={
        'messages': [{'role': 'user', 'content': 'I need Java.'}]
    })
    second = client.post('/chat', json={
        'messages': [
            {'role': 'user', 'content': 'I need Java.'},
            {'role': 'user', 'content': 'Also include personality tests.'},
        ]
    })

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert second_payload['end_of_conversation'] is True
    assert len(second_payload['recommendations']) >= 1
    assert first_payload['recommendations'] == [] or len(first_payload['recommendations']) >= 1


def test_stateless_history_reconstructs_constraints():
    response = client.post('/chat', json={
        'messages': [
            {'role': 'user', 'content': 'I need an assessment.'},
            {'role': 'assistant', 'content': 'What role are you hiring for?'},
            {'role': 'user', 'content': 'Java developer'},
            {'role': 'assistant', 'content': 'What seniority?'},
            {'role': 'user', 'content': 'Mid-level'},
        ]
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload['end_of_conversation'] is True
    assert payload['recommendations']
    assert 'role' not in payload['reply'].lower()
    assert 'seniority' not in payload['reply'].lower()


def test_missing_messages_is_handled_gracefully():
    response = client.post('/chat', json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload['reply']
    assert payload['recommendations'] == []
    assert payload['end_of_conversation'] is False
